using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Web.WebView2.WinForms;
using Microsoft.Web.WebView2.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SkinSight.Models;
using SkinSight.Services;

// ═══════════════════════════════════════════════════════════════════
// SkinSight Desktop – Windows App (WinForms + Edge WebView2)
// ═══════════════════════════════════════════════════════════════════
//
//   ┌─────────────────────────────────────────────┐
//   │  SkinSight.exe (single process)             │
//   │                                             │
//   │  ┌──────────────┐   ┌────────────────────┐  │
//   │  │  WinForms     │   │  Kestrel (in-proc) │  │
//   │  │  + WebView2   │──▶│  ASP.NET Core API  │  │
//   │  │  (Edge engine)│   │  port: auto        │  │
//   │  └──────────────┘   └────────────────────┘  │
//   │                                             │
//   │  • Uses Edge WebView2 (built into Win10/11) │
//   │  • ~20MB RAM for UI (vs 300MB+ Chrome)      │
//   │  • Single .exe distribution                 │
//   └─────────────────────────────────────────────┘
//
// ═══════════════════════════════════════════════════════════════════

namespace SkinSight;

static class Program
{
    [STAThread]
    static void Main()
    {
        // Find a free port
        var port = GetFreePort();
        var serverUrl = $"http://localhost:{port}";

        // Start Kestrel backend on a background thread
        var serverReady = new ManualResetEventSlim(false);
        var serverThread = new Thread(() => StartServer(port, serverReady))
        {
            IsBackground = true,
            Name = "KestrelServer"
        };
        serverThread.Start();

        // Wait for server to be ready
        if (!serverReady.Wait(TimeSpan.FromSeconds(30)))
        {
            MessageBox.Show("Backend server could not start in 30 seconds.",
                "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        // Launch WinForms app with WebView2
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        Application.Run(new MainForm(serverUrl));
    }

    static int GetFreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    // ═══════════════════════════════════════════════════════════════
    // Kestrel web server (identical API to the web version)
    // ═══════════════════════════════════════════════════════════════

    static void StartServer(int listenPort, ManualResetEventSlim readySignal)
    {
        var exeDir = AppContext.BaseDirectory;
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            ContentRootPath = exeDir,
            WebRootPath = Path.Combine(exeDir, "wwwroot")
        });

        var config = new SkinSightConfig();

        builder.Services.AddSingleton(config);
        builder.Services.AddSingleton<SlideCache>();
        builder.Services.AddSingleton<AnalysisService>();

        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Listen(IPAddress.Loopback, listenPort);
            options.Limits.MaxRequestBodySize = (long)config.MaxUploadSizeGB * 1024 * 1024 * 1024;
        });

        // Increase multipart form body limit (default ~128MB is too small for large WSI files)
        builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
        {
            options.MultipartBodyLengthLimit = (long)config.MaxUploadSizeGB * 1024 * 1024 * 1024;
        });

        builder.Services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
            options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
        });

        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);
        builder.Logging.AddFilter("SkinSight", LogLevel.Information);

        var app = builder.Build();
        app.UseStaticFiles();

        var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("SkinSight");
        var analysisService = app.Services.GetRequiredService<AnalysisService>();
        var slideCache = app.Services.GetRequiredService<SlideCache>();

        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        // ═══════════════════════════════════════════════════════════
        // Routes
        // ═══════════════════════════════════════════════════════════

        app.MapGet("/", () => Results.File(
            Path.Combine(app.Environment.WebRootPath, "index.html"), "text/html"));

        // ── Models ──
        var modelRegistry = new List<ModelInfo>
        {
            new() { Key = "phikon",     Name = "Phikon",          Display = "Phikon (Pathology Foundation)",      F1 = 0.9250, Auc = 0.9811, Description = "Pathology-specialized ViT, highest F1 (92.5%)",    Available = true },
            new() { Key = "convnext_s", Name = "ConvNeXt-Small",  Display = "ConvNeXt-Small",             F1 = 0.8716, Auc = 0.9551, Description = "Modern CNN, strong performer (87.2% F1)",          Available = true },
            new() { Key = "convnext_b", Name = "ConvNeXt-Base",   Display = "ConvNeXt-Base",              F1 = 0.8681, Auc = 0.9663, Description = "Larger ConvNeXt, high AUC (96.6%)",              Available = true },
            new() { Key = "dinov2",     Name = "DINOv2-Base",     Display = "DINOv2-Base (Self-Supervised)", F1 = 0.8198, Auc = 0.9477, Description = "Self-supervised ViT from Meta (82.0% F1)",   Available = true },
            new() { Key = "resnet18",   Name = "ResNet18",       Display = "ResNet18 (Baseline)",       F1 = 0.8155, Auc = 0.9414, Description = "Lightweight baseline CNN (81.6% F1)",     Available = true },
            new() { Key = "resnet50",   Name = "ResNet50",       Display = "ResNet50",       F1 = 0.7988, Auc = 0.9404, Description = "Deeper ResNet (79.9% F1)",         Available = true },
        };

        app.MapGet("/api/models", () =>
        {
            var response = new ModelsResponse
            {
                Models = modelRegistry,
                Default = "phikon",
                Ensemble = new EnsembleInfo
                {
                    Key = "ensemble",
                    Name = "Ensemble",
                    Display = "Ensemble (Top-3 Models)",
                    Description = "Averages Phikon + ConvNeXt-Small + ConvNeXt-Base",
                    Models = new List<string> { "phikon", "convnext_s", "convnext_b" },
                }
            };
            return Results.Json(response, jsonOptions);
        });

        // ── Upload ──
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
                await file.CopyToAsync(stream);

            // Read model selection
            var modelKey = form.ContainsKey("model") ? form["model"].ToString() : "phikon";
            var modelInfo = modelRegistry.FirstOrDefault(m => m.Key == modelKey);
            var modelDisplay = modelKey == "ensemble"
                ? "Ensemble (Top-3 Models)"
                : modelInfo?.Display ?? modelKey;

            var fileSizeMb = Math.Round(new FileInfo(slidePath).Length / (1024.0 * 1024.0), 1);
            logger.LogInformation("Uploaded {Filename} ({SizeMb} MB) -> {JobId} [model: {Model}]",
                file.FileName, fileSizeMb, jobId, modelKey);

            analysisService.Jobs[jobId] = new AnalysisJob
            {
                JobId = jobId, Status = "queued", Progress = 0,
                Message = "Upload complete. Starting analysis...",
                Filename = file.FileName, SlidePath = slidePath,
                FileSizeMb = fileSizeMb,
                ModelKey = modelKey,
                ModelDisplay = modelDisplay,
                CreatedAt = DateTime.Now,
            };

            analysisService.StartAnalysis(jobId, slidePath);
            return Results.Ok(new UploadResponse
                { JobId = jobId, Filename = file.FileName, SizeMb = fileSizeMb, Model = modelDisplay });
        });

        // ── Status ──
        app.MapGet("/api/status/{jobId}", (string jobId) =>
        {
            if (!analysisService.Jobs.TryGetValue(jobId, out var job))
                return Results.NotFound(new { error = "Job not found" });
            var response = new JobStatusResponse
            {
                JobId = job.JobId, Status = job.Status, Progress = job.Progress,
                Message = job.Message, Filename = job.Filename,
                FileSizeMb = job.FileSizeMb, ModelDisplay = job.ModelDisplay,
                CreatedAt = job.CreatedAt.ToString("o"),
                SlideInfo = job.SlideInfo, Result = job.Result,
            };
            return Results.Json(response, jsonOptions);
        });

        // ── DZI ──
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

        app.MapGet("/api/results/{jobId}/dzi/slide_files/{level}/{tileName}",
            (string jobId, int level, string tileName) =>
        {
            var slidePath = GetSlidePath(jobId);
            if (slidePath == null) return Results.NotFound();
            var nameWithoutExt = Path.GetFileNameWithoutExtension(tileName);
            var parts = nameWithoutExt.Split('_');
            if (parts.Length != 2 || !int.TryParse(parts[0], out int col)
                                  || !int.TryParse(parts[1], out int row))
                return Results.BadRequest("Invalid tile name");
            var (slide, dz) = slideCache.Get(slidePath);
            if (level < 0 || level >= dz.LevelCount) return Results.NotFound();
            var (cols, rows) = dz.LevelTiles[level];
            if (col < 0 || col >= cols || row < 0 || row >= rows) return Results.NotFound();
            using var tile = dz.GetTile(level, col, row);
            var ms = new MemoryStream();
            tile.SaveAsJpeg(ms, new JpegEncoder { Quality = config.DziQuality });
            ms.Position = 0;
            return Results.File(ms, "image/jpeg");
        });

        // ── Result assets ──
        app.MapGet("/api/results/{jobId}/heatmap", (string jobId) =>
        {
            var p = Path.Combine(analysisService.ResultsDir, jobId, "heatmap.jpg");
            return File.Exists(p) ? Results.File(p, "image/jpeg") : Results.NotFound();
        });

        app.MapGet("/api/results/{jobId}/heatmap_only", (string jobId) =>
        {
            var p = Path.Combine(analysisService.ResultsDir, jobId, "heatmap_only.png");
            return File.Exists(p) ? Results.File(p, "image/png") : Results.NotFound();
        });

        app.MapGet("/api/results/{jobId}/thumbnail", (string jobId) =>
        {
            var p = Path.Combine(analysisService.ResultsDir, jobId, "thumbnail.jpg");
            return File.Exists(p) ? Results.File(p, "image/jpeg") : Results.NotFound();
        });

        app.MapGet("/api/results/{jobId}/tiles/{filename}", (string jobId, string filename) =>
        {
            var p = Path.Combine(analysisService.ResultsDir, jobId, "tiles", filename);
            return File.Exists(p) ? Results.File(p, "image/jpeg") : Results.NotFound();
        });

        // ── History ──
        app.MapGet("/api/history", () =>
        {
            analysisService.CleanupOldResults();
            var history = analysisService.Jobs.Values
                .OrderByDescending(j => j.CreatedAt)
                .Select(j => new HistoryItem
                {
                    JobId = j.JobId, Filename = j.Filename, Status = j.Status,
                    Model = j.ModelDisplay,
                    CreatedAt = j.CreatedAt.ToString("o"), Result = j.Result,
                }).ToList();
            return Results.Json(history, jsonOptions);
        });

        // ── Export ──
        app.MapGet("/api/results/{jobId}/export", (string jobId) =>
        {
            if (!analysisService.Jobs.TryGetValue(jobId, out var job) || job.Result == null)
                return Results.NotFound(new { error = "No results available" });
            var exportData = new
            {
                job_id = jobId, filename = job.Filename,
                analysis_date = job.CreatedAt.ToString("o"),
                result = job.Result, slide_info = job.SlideInfo,
            };
            var exportPath = Path.Combine(analysisService.ResultsDir, jobId, "export.json");
            Directory.CreateDirectory(Path.GetDirectoryName(exportPath)!);
            System.IO.File.WriteAllText(exportPath, JsonSerializer.Serialize(exportData, jsonOptions));
            return Results.File(exportPath, "application/json", $"skinsight_report_{jobId}.json");
        });

        // ── Delete ──
        app.MapPost("/api/results/{jobId}/delete", (string jobId) =>
        {
            foreach (var dir in new[] {
                Path.Combine(analysisService.ResultsDir, jobId),
                Path.Combine(analysisService.UploadDir, jobId) })
            {
                if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
            }
            if (analysisService.Jobs.TryRemove(jobId, out var job2))
                slideCache.Remove(job2.SlidePath);
            return Results.Ok(new { deleted = jobId });
        });

        // ── Demo ──
        app.MapGet("/api/demo", () =>
        {
            var projectDir = Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, ".."));
            var resultsPath = Path.Combine(projectDir, "data", "mil_results", "mil_results.json");
            if (!System.IO.File.Exists(resultsPath))
                return Results.NotFound(new { error = "No demo results available" });
            var json = System.IO.File.ReadAllText(resultsPath);
            var doc = JsonDocument.Parse(json);
            return Results.Json(doc.RootElement);
        });

        // ── Info ──
        app.MapGet("/api/info", () =>
        {
            long uploadSize = Directory.Exists(analysisService.UploadDir)
                ? new DirectoryInfo(analysisService.UploadDir)
                    .EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length) : 0;
            long resultsSize = Directory.Exists(analysisService.ResultsDir)
                ? new DirectoryInfo(analysisService.ResultsDir)
                    .EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length) : 0;
            return Results.Json(new ServerInfoResponse
            {
                Status = "ok", Jobs = analysisService.Jobs.Count,
                UploadsMb = Math.Round(uploadSize / 1e6, 1),
                ResultsMb = Math.Round(resultsSize / 1e6, 1),
                NModels = modelRegistry.Count,
                Classes = ClassInfo.Names.Values.ToList(),
            }, jsonOptions);
        });

        // ── Helper ──
        string? GetSlidePath(string jobId)
        {
            if (!analysisService.Jobs.TryGetValue(jobId, out var job)) return null;
            if (!string.IsNullOrEmpty(job.SlidePath) && System.IO.File.Exists(job.SlidePath))
                return job.SlidePath;
            return null;
        }

        logger.LogInformation("SkinSight Desktop backend on port {Port}", listenPort);

        // Start listening, then signal ready
        app.Start();
        readySignal.Set();

        // Block until app is stopped
        app.WaitForShutdown();
    }
}

// ═══════════════════════════════════════════════════════════════════
// MainForm - WinForms window with embedded Edge WebView2
// ═══════════════════════════════════════════════════════════════════

public class MainForm : Form
{
    private readonly WebView2 _webView;
    private readonly string _serverUrl;

    public MainForm(string serverUrl)
    {
        _serverUrl = serverUrl;

        // Form settings
        Text = "SkinSight – AI Skin Cancer WSI Analysis";
        Width = 1400;
        Height = 900;
        MinimumSize = new System.Drawing.Size(900, 600);
        StartPosition = FormStartPosition.CenterScreen;
        Icon = SystemIcons.Application;

        // WebView2
        _webView = new WebView2
        {
            Dock = DockStyle.Fill
        };

        _webView.CoreWebView2InitializationCompleted += OnWebViewReady;
        Controls.Add(_webView);
    }

    protected override async void OnLoad(EventArgs e)
    {
        base.OnLoad(e);

        try
        {
            // Initialize WebView2 with user data folder in temp
            var env = await CoreWebView2Environment.CreateAsync(
                userDataFolder: Path.Combine(Path.GetTempPath(), "SkinSight_WebView2"));
            await _webView.EnsureCoreWebView2Async(env);
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                $"Edge WebView2 could not be initialized.\n\n{ex.Message}\n\nMake sure Microsoft Edge is installed.",
                "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            Close();
        }
    }

    private void OnWebViewReady(object? sender, CoreWebView2InitializationCompletedEventArgs e)
    {
        if (!e.IsSuccess)
        {
            MessageBox.Show(
                $"WebView2 init failed: {e.InitializationException?.Message}",
                "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            Close();
            return;
        }

        // Navigate to the embedded server
        _webView.CoreWebView2.Navigate(_serverUrl + "/");
    }
}
