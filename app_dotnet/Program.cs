using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.FileProviders;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SkinSight.Models;
using SkinSight.Services;

// ═══════════════════════════════════════════════════════════════════
// SkinSight – ASP.NET Core Minimal API  (.NET 8 port of Flask server)
// ═══════════════════════════════════════════════════════════════════
//
// This is a faithful port of the Python Flask server (app/server.py)
// to .NET 8, using:
//   • OpenSlide via P/Invoke (libopenslide native library)
//   • SixLabors.ImageSharp for image processing
//   • ONNX Runtime for ML inference (instead of PyTorch)
//
// PREREQUISITES:
//   1. .NET 8 SDK: https://dot.net/download
//   2. OpenSlide native library:
//        Ubuntu:  sudo apt-get install libopenslide0 libopenslide-dev
//        macOS:   brew install openslide
//   3. (Optional) ONNX model files for real inference
//
// RUN:
//   cd app_dotnet
//   dotnet run
//
// ═══════════════════════════════════════════════════════════════════

var builder = WebApplication.CreateBuilder(args);

// ─── Configuration ──────────────────────────────────────────────
var config = builder.Configuration.GetSection("SkinSight").Get<SkinSightConfig>()
             ?? new SkinSightConfig();

// ─── Services ───────────────────────────────────────────────────
builder.Services.AddSingleton(config);
builder.Services.AddSingleton<SlideCache>();
builder.Services.AddSingleton<AnalysisService>();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

// Allow large uploads
builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = (long)config.MaxUploadSizeGB * 1024 * 1024 * 1024;
});

// Increase multipart form body limit (default ~128MB is too small for large WSI files)
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = (long)config.MaxUploadSizeGB * 1024 * 1024 * 1024;
});

// JSON serialization: camelCase
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

var app = builder.Build();

app.UseCors();
app.UseStaticFiles();   // Serve wwwroot/
app.UseStaticFiles(new StaticFileOptions
{
    RequestPath = "/static",
    FileProvider = new PhysicalFileProvider(app.Environment.WebRootPath)
});

var logger = app.Services.GetRequiredService<ILogger<Program>>();
var analysisService = app.Services.GetRequiredService<AnalysisService>();
var slideCache = app.Services.GetRequiredService<SlideCache>();

var jsonOptions = new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
};

ModelInfo CloneModelForResponse(ModelInfo m)
{
    var threshold = ModelCatalog.BuildThresholdPolicy(m.Key);
    return new ModelInfo
    {
        Key = m.Key,
        Name = m.Name,
        Display = m.Display,
        Group = m.Group,
        F1 = m.F1,
        Auc = m.Auc,
        MelFn = m.MelFn,
        Description = m.Description,
        Available = m.Available,
        ThresholdPolicy = threshold,
        ThresholdLabel = threshold.TryGetValue("label", out var label) ? label?.ToString() : null,
    };
}

EnsembleInfo CloneEnsembleForResponse(EnsembleInfo e)
{
    var threshold = ModelCatalog.BuildThresholdPolicy(e.Key);
    return new EnsembleInfo
    {
        Key = e.Key,
        Name = e.Name,
        Display = e.Display,
        Description = e.Description,
        Models = e.Models.ToList(),
        F1 = e.F1,
        Auc = e.Auc,
        MelFn = e.MelFn,
        Gated = e.Gated,
        GatingPolicy = e.GatingPolicy == null ? null : new Dictionary<string, object?>(e.GatingPolicy),
        ThresholdPolicy = threshold,
        ThresholdLabel = threshold.TryGetValue("label", out var label) ? label?.ToString() : null,
    };
}

// ═══════════════════════════════════════════════════════════════
//  ROUTES
// ═══════════════════════════════════════════════════════════════

// ─── Static / SPA ──────────────────────────────────────────────

app.MapGet("/", () => Results.File(
    Path.Combine(app.Environment.WebRootPath, "index.html"),
    "text/html"));

// ─── Models ────────────────────────────────────────────────────

app.MapGet("/api/models", () =>
{
    var models = ModelCatalog.Models
        .OrderByDescending(m => m.F1)
        .Select(CloneModelForResponse)
        .ToList();
    var ensembles = ModelCatalog.Ensembles
        .Select(CloneEnsembleForResponse)
        .ToList();

    var response = new ModelsResponse
    {
        Models = models,
        Ensembles = ensembles,
        Ensemble = ensembles.FirstOrDefault(e => e.Key == "ensemble_3_best"),
        Default = ModelCatalog.DefaultModelKey,
    };
    return Results.Json(response, jsonOptions);
});

// ─── Upload ────────────────────────────────────────────────────

app.MapPost("/api/upload", async (HttpRequest request) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "No file uploaded" });

    var form = await request.ReadFormAsync();
    var file = form.Files["slide"];
    if (file == null || string.IsNullOrEmpty(file.FileName))
        return Results.BadRequest(new { error = "No file uploaded" });

    var ext = Path.GetExtension(file.FileName).ToLower();
    var allowedExts = new[] { ".tif", ".tiff", ".svs", ".ndpi", ".mrxs", ".scn" };
    if (!allowedExts.Contains(ext))
        return Results.BadRequest(new { error = $"Unsupported format: {ext}" });

    var jobId = Guid.NewGuid().ToString()[..8];
    var slideDir = Path.Combine(analysisService.UploadDir, jobId);
    Directory.CreateDirectory(slideDir);
    var slidePath = Path.Combine(slideDir, file.FileName);

    await using (var stream = new FileStream(slidePath, FileMode.Create))
    {
        await file.CopyToAsync(stream);
    }

    // Read model selection
    var modelKey = form.ContainsKey("model") ? form["model"].ToString() : ModelCatalog.DefaultModelKey;
    if (!ModelCatalog.IsKnownKey(modelKey))
        modelKey = ModelCatalog.DefaultModelKey;
    var modelDisplay = ModelCatalog.DisplayFor(modelKey);

    var fileSizeMb = Math.Round(new FileInfo(slidePath).Length / (1024.0 * 1024.0), 1);
    logger.LogInformation("Uploaded {Filename} ({SizeMb} MB) -> {JobId} [model: {Model}]",
        file.FileName, fileSizeMb, jobId, modelKey);

    analysisService.Jobs[jobId] = new AnalysisJob
    {
        JobId = jobId,
        Status = "queued",
        Progress = 0,
        Message = "Upload complete. Starting analysis...",
        Filename = file.FileName,
        SlidePath = slidePath,
        FileSizeMb = fileSizeMb,
        ModelKey = modelKey,
        ModelDisplay = modelDisplay,
        CreatedAt = DateTime.Now,
    };

    analysisService.StartAnalysis(jobId, slidePath);

    return Results.Ok(new UploadResponse
    {
        JobId = jobId,
        Filename = file.FileName,
        SizeMb = fileSizeMb,
        Model = modelDisplay,
    });
});

// ─── Status ────────────────────────────────────────────────────

app.MapGet("/api/status/{jobId}", (string jobId) =>
{
    if (!analysisService.Jobs.TryGetValue(jobId, out var job))
        return Results.NotFound(new { error = "Job not found" });

    var response = new JobStatusResponse
    {
        JobId = job.JobId,
        Status = job.Status,
        Progress = job.Progress,
        Message = job.Message,
        Filename = job.Filename,
        FileSizeMb = job.FileSizeMb,
        ModelDisplay = job.ModelDisplay,
        CreatedAt = job.CreatedAt.ToString("o"),
        SlideInfo = job.SlideInfo,
        Result = job.Result,
    };
    return Results.Json(response, jsonOptions);
});

// ─── DZI Descriptor ────────────────────────────────────────────

app.MapGet("/api/results/{jobId}/dzi/slide.dzi", (string jobId) =>
{
    var slidePath = GetSlidePath(jobId);
    if (slidePath == null) return Results.NotFound();

    var (slide, dz) = slideCache.Get(slidePath);
    var (w, h) = dz.LevelDimensions[^1];

    var xml = $"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
               Format="jpeg" Overlap="{config.DziOverlap}" TileSize="{config.DziTileSize}">
          <Size Width="{w}" Height="{h}"/>
        </Image>
        """;
    return Results.Content(xml, "application/xml");
});

// ─── DZI Tile ──────────────────────────────────────────────────

app.MapGet("/api/results/{jobId}/dzi/slide_files/{level}/{tileName}",
    (string jobId, int level, string tileName) =>
{
    var slidePath = GetSlidePath(jobId);
    if (slidePath == null) return Results.NotFound();

    // Parse "col_row.jpeg"
    var nameWithoutExt = Path.GetFileNameWithoutExtension(tileName);
    var parts = nameWithoutExt.Split('_');
    if (parts.Length != 2 ||
        !int.TryParse(parts[0], out int col) ||
        !int.TryParse(parts[1], out int row))
        return Results.BadRequest("Invalid tile name");

    var (slide, dz) = slideCache.Get(slidePath);

    if (level < 0 || level >= dz.LevelCount)
        return Results.NotFound();
    var (cols, rows) = dz.LevelTiles[level];
    if (col < 0 || col >= cols || row < 0 || row >= rows)
        return Results.NotFound();

    using var tile = dz.GetTile(level, col, row);
    var ms = new MemoryStream();
    tile.SaveAsJpeg(ms, new JpegEncoder { Quality = config.DziQuality });
    ms.Position = 0;
    return Results.File(ms, "image/jpeg");
});

// ─── Result Assets ─────────────────────────────────────────────

app.MapGet("/api/results/{jobId}/heatmap", (string jobId) =>
{
    var path = Path.Combine(analysisService.ResultsDir, jobId, "heatmap.jpg");
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/results/{jobId}/heatmap/{variant}", (string jobId, string variant) =>
{
    var specific = Path.Combine(analysisService.ResultsDir, jobId, $"{variant}_heatmap.jpg");
    var fallback = Path.Combine(analysisService.ResultsDir, jobId, "heatmap.jpg");
    var path = File.Exists(specific) ? specific : fallback;
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/results/{jobId}/heatmap_only", (string jobId) =>
{
    var path = Path.Combine(analysisService.ResultsDir, jobId, "heatmap_only.png");
    return File.Exists(path)
        ? Results.File(path, "image/png")
        : Results.NotFound();
});

app.MapGet("/api/results/{jobId}/heatmap_only/{variant}", (string jobId, string variant) =>
{
    var specific = Path.Combine(analysisService.ResultsDir, jobId, $"{variant}_heatmap_only.png");
    var fallback = Path.Combine(analysisService.ResultsDir, jobId, "heatmap_only.png");
    var path = File.Exists(specific) ? specific : fallback;
    return File.Exists(path)
        ? Results.File(path, "image/png")
        : Results.NotFound();
});

app.MapGet("/api/results/{jobId}/thumbnail", (string jobId) =>
{
    var path = Path.Combine(analysisService.ResultsDir, jobId, "thumbnail.jpg");
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/results/{jobId}/tiles/{filename}", (string jobId, string filename) =>
{
    var path = Path.Combine(analysisService.ResultsDir, jobId, "tiles", filename);
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/retrieval/thumbnails/{slideId}.jpg", (string slideId) =>
{
    var projectDir = Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, ".."));
    var path = Path.Combine(projectDir, "results", "phase4_retrieval", "thumbnails", $"{slideId}.jpg");
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/retrieval/continual/thumbnails/{jobId}.jpg", (string jobId) =>
{
    var path = Path.Combine(analysisService.ResultsDir, jobId, "thumbnail.jpg");
    return File.Exists(path)
        ? Results.File(path, "image/jpeg")
        : Results.NotFound();
});

app.MapGet("/api/retrieval/cases/{slideId}/compare", (string slideId) =>
{
    return Results.NotFound(new
    {
        error = "Retrieval case comparison is implemented in the Flask runtime. The .NET port serves the updated UI and compatibility retrieval summaries."
    });
});

// ─── History ───────────────────────────────────────────────────

app.MapGet("/api/history", () =>
{
    analysisService.CleanupOldResults();

    var history = analysisService.Jobs.Values
        .OrderByDescending(j => j.CreatedAt)
        .Select(j => new HistoryItem
        {
            JobId = j.JobId,
            Filename = j.Filename,
            Status = j.Status,
            Model = j.ModelDisplay,
            CreatedAt = j.CreatedAt.ToString("o"),
            Result = j.Result,
        })
        .ToList();
    return Results.Json(history, jsonOptions);
});

// ─── Export ────────────────────────────────────────────────────

app.MapGet("/api/results/{jobId}/export", (string jobId) =>
{
    if (!analysisService.Jobs.TryGetValue(jobId, out var job) || job.Result == null)
        return Results.NotFound(new { error = "No results available" });

    var exportData = new
    {
        job_id = jobId,
        filename = job.Filename,
        analysis_date = job.CreatedAt.ToString("o"),
        result = job.Result,
        slide_info = job.SlideInfo,
    };

    var exportPath = Path.Combine(analysisService.ResultsDir, jobId, "export.json");
    Directory.CreateDirectory(Path.GetDirectoryName(exportPath)!);
    File.WriteAllText(exportPath,
        JsonSerializer.Serialize(exportData, jsonOptions));

    return Results.File(exportPath, "application/json",
        $"skinsight_report_{jobId}.json");
});

app.MapGet("/api/results/{jobId}/report.pdf", (string jobId) =>
{
    var pdfPath = analysisService.BuildPdfReport(jobId);
    return pdfPath != null && File.Exists(pdfPath)
        ? Results.File(pdfPath, "application/pdf", $"skinsight_report_{jobId}.pdf")
        : Results.NotFound(new { error = "No PDF report available" });
});

// ─── Delete ────────────────────────────────────────────────────

app.MapPost("/api/results/{jobId}/delete", (string jobId) =>
{
    foreach (var dir in new[] {
        Path.Combine(analysisService.ResultsDir, jobId),
        Path.Combine(analysisService.UploadDir, jobId) })
    {
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
    }

    if (analysisService.Jobs.TryRemove(jobId, out var job))
        slideCache.Remove(job.SlidePath);

    return Results.Ok(new { deleted = jobId });
});

// ─── Demo ──────────────────────────────────────────────────────

app.MapGet("/api/demo", () =>
{
    var projectDir = Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, ".."));
    var resultsPath = Path.Combine(projectDir, "data", "mil_results", "mil_results.json");

    if (!File.Exists(resultsPath))
        return Results.NotFound(new { error = "No demo results available" });

    var json = File.ReadAllText(resultsPath);
    var doc = JsonDocument.Parse(json);
    return Results.Json(doc.RootElement);
});

// ─── Info ──────────────────────────────────────────────────────

app.MapGet("/api/info", () =>
{
    long uploadSize = Directory.Exists(analysisService.UploadDir)
        ? new DirectoryInfo(analysisService.UploadDir)
            .EnumerateFiles("*", SearchOption.AllDirectories)
            .Sum(f => f.Length)
        : 0;

    long resultsSize = Directory.Exists(analysisService.ResultsDir)
        ? new DirectoryInfo(analysisService.ResultsDir)
            .EnumerateFiles("*", SearchOption.AllDirectories)
            .Sum(f => f.Length)
        : 0;

    return Results.Json(new ServerInfoResponse
    {
        Status = "ok",
        Jobs = analysisService.Jobs.Count,
        UploadsMb = Math.Round(uploadSize / 1e6, 1),
        ResultsMb = Math.Round(resultsSize / 1e6, 1),
        NModels = ModelCatalog.Models.Count,
        Classes = ClassInfo.Names.Values.ToList(),
        RetrievalBanks = new List<string> { "phase4_reference_bank", "dotnet_compatibility_summary" },
    }, jsonOptions);
});

// ═══════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════

string? GetSlidePath(string jobId)
{
    if (!analysisService.Jobs.TryGetValue(jobId, out var job)) return null;
    if (!string.IsNullOrEmpty(job.SlidePath) && File.Exists(job.SlidePath))
        return job.SlidePath;
    return null;
}

// ═══════════════════════════════════════════════════════════════
//  START
// ═══════════════════════════════════════════════════════════════

var port = builder.Configuration.GetValue("PORT", 5001);
logger.LogInformation("Starting SkinSight .NET server on port {Port}", port);
logger.LogInformation("  Delete slide after analysis: {DeleteAfterAnalysis}",
    config.DeleteSlideAfterAnalysis);
logger.LogInformation("  Result retention: {RetentionMinutes} min",
    config.ResultRetentionMinutes);

app.Run($"http://0.0.0.0:{port}");
