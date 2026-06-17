using System.Net;
using System.Net.Sockets;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
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
    private static readonly object StartupLogLock = new();

    [STAThread]
    static void Main()
    {
        File.WriteAllText(StartupLogPath(), $"SkinSight desktop startup {DateTime.Now:o}{Environment.NewLine}");
        Application.ThreadException += (_, e) => LogStartupError("UI thread exception", e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            if (e.ExceptionObject is Exception ex)
                LogStartupError("Unhandled exception", ex);
        };

        // Find a free port
        var port = GetFreePort();
        var serverUrl = $"http://localhost:{port}";
        Process? externalBackend = null;
        var backendMode = Environment.GetEnvironmentVariable("SKINSIGHT_BACKEND_MODE") ?? "python";

        if (backendMode.Equals("python", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                externalBackend = StartPythonBackend(port);
                if (!WaitForBackend(serverUrl, TimeSpan.FromSeconds(120), externalBackend))
                    throw new TimeoutException($"Python backend did not become ready at {serverUrl}.");
                LogStartupInfo($"Python backend ready at {serverUrl}");
            }
            catch (Exception ex)
            {
                LogStartupError("Python backend failed to start", ex);
                TryKillProcessTree(externalBackend);
                MessageBox.Show(
                    $"Python backend failed to start.\n\n{ex.Message}\n\nLog: {StartupLogPath()}",
                    "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }
        else
        {
            Exception? serverStartupError = null;

            // Start Kestrel backend on a background thread.
            using var serverReady = new ManualResetEventSlim(false);
            var serverThread = new Thread(() =>
            {
                try
                {
                    StartServer(port, serverReady);
                }
                catch (Exception ex)
                {
                    serverStartupError = ex;
                    LogStartupError("Backend server failed before ready", ex);
                    serverReady.Set();
                }
            })
            {
                IsBackground = true,
                Name = "KestrelServer"
            };
            serverThread.Start();

            // Wait for server to be ready.
            if (!serverReady.Wait(TimeSpan.FromSeconds(30)))
            {
                MessageBox.Show("Backend server could not start in 30 seconds.",
                    "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            if (serverStartupError != null)
            {
                MessageBox.Show(
                    $"Backend server failed to start.\n\n{serverStartupError.Message}\n\nLog: {StartupLogPath()}",
                    "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        // Launch WinForms app with WebView2
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);
        LogStartupInfo($"Launching WebView2 at {serverUrl}");
        Application.Run(new MainForm(serverUrl, externalBackend));
    }

    static int GetFreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    static string StartupLogPath() => Path.Combine(Path.GetTempPath(), "skinsight_desktop_startup.log");

    static void LogStartupInfo(string message)
    {
        AppendStartupLog($"INFO  {DateTime.Now:o} {message}{Environment.NewLine}");
    }

    static void LogStartupError(string message, Exception ex)
    {
        AppendStartupLog($"ERROR {DateTime.Now:o} {message}{Environment.NewLine}{ex}{Environment.NewLine}");
    }

    static void AppendStartupLog(string text)
    {
        lock (StartupLogLock)
        {
            File.AppendAllText(StartupLogPath(), text);
        }
    }

    static Process StartPythonBackend(int port)
    {
        var distro = Environment.GetEnvironmentVariable("SKINSIGHT_WSL_DISTRO") ?? "Ubuntu-22.04";
        var projectRoot = Environment.GetEnvironmentVariable("SKINSIGHT_WSL_PROJECT_ROOT") ?? "/home/byalc/phase1_project";
        var envRoot = Environment.GetEnvironmentVariable("SKINSIGHT_PYTHON_ENV") ?? "/home/byalc/phase1_env";
        var bashCommand =
            $"cd {ShellQuote(projectRoot)} && " +
            $"source {ShellQuote(Path.Combine(envRoot, "bin", "activate").Replace("\\", "/"))} && " +
            $"export PORT={port} PYTHONUNBUFFERED=1 && " +
            "python app/server.py";

        LogStartupInfo($"Starting Python backend via WSL distro={distro}, project={projectRoot}, env={envRoot}, port={port}");

        var psi = new ProcessStartInfo("wsl.exe")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add("-d");
        psi.ArgumentList.Add(distro);
        psi.ArgumentList.Add("--");
        psi.ArgumentList.Add("bash");
        psi.ArgumentList.Add("-lc");
        psi.ArgumentList.Add(bashCommand);

        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                LogStartupInfo($"python stdout: {e.Data}");
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                LogStartupInfo($"python stderr: {e.Data}");
        };
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return process;
    }

    static bool WaitForBackend(string serverUrl, TimeSpan timeout, Process? backendProcess)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (backendProcess != null && backendProcess.HasExited)
            {
                LogStartupInfo($"Backend process exited early with code {backendProcess.ExitCode}");
                return false;
            }

            try
            {
                using var response = client.GetAsync($"{serverUrl}/api/info").GetAwaiter().GetResult();
                if ((int)response.StatusCode < 500)
                    return true;
            }
            catch
            {
                // Server is still starting. Keep polling.
            }

            Thread.Sleep(1000);
        }
        return false;
    }

    static string ShellQuote(string value) => "'" + value.Replace("'", "'\"'\"'") + "'";

    static void TryKillProcessTree(Process? process)
    {
        try
        {
            if (process != null && !process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Best-effort cleanup only.
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Kestrel web server (identical API to the web version)
    // ═══════════════════════════════════════════════════════════════

    static string FindProjectRoot(string startPath)
    {
        var envRoot = Environment.GetEnvironmentVariable("SKINSIGHT_PROJECT_ROOT");
        if (!string.IsNullOrWhiteSpace(envRoot) &&
            Directory.Exists(Path.Combine(envRoot, "app_desktop")) &&
            Directory.Exists(Path.Combine(envRoot, "app_dotnet")))
            return Path.GetFullPath(envRoot);

        var dir = new DirectoryInfo(Path.GetFullPath(startPath));
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "app_desktop")) &&
                Directory.Exists(Path.Combine(dir.FullName, "app_dotnet")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return Path.GetFullPath(Path.Combine(startPath, "..", "..", "..", "..", ".."));
    }
    static void StartServer(int listenPort, ManualResetEventSlim readySignal)
    {
        var exeDir = AppContext.BaseDirectory;
        var projectRoot = FindProjectRoot(exeDir);
        LogStartupInfo($"Executable directory: {exeDir}");
        LogStartupInfo($"Project root: {projectRoot}");
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            ContentRootPath = exeDir,
            WebRootPath = Path.Combine(exeDir, "wwwroot")
        });
        LogStartupInfo($"Content root: {exeDir}");
        LogStartupInfo($"Web root: {Path.Combine(exeDir, "wwwroot")}");

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
        app.UseStaticFiles(new StaticFileOptions
        {
            RequestPath = "/static",
            FileProvider = new PhysicalFileProvider(app.Environment.WebRootPath)
        });

        var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("SkinSight");
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
            var projectDir = FindProjectRoot(AppContext.BaseDirectory);
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
            var projectDir = FindProjectRoot(AppContext.BaseDirectory);
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

        logger.LogInformation("SkinSight Desktop backend on port {Port}", listenPort);

        // Start listening, then signal ready
        app.Start();
        LogStartupInfo($"Backend listening on http://localhost:{listenPort}");
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
    private readonly Process? _backendProcess;

    public MainForm(string serverUrl, Process? backendProcess = null)
    {
        _serverUrl = serverUrl;
        _backendProcess = backendProcess;

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
            File.AppendAllText(Path.Combine(Path.GetTempPath(), "skinsight_desktop_startup.log"),
                $"ERROR {DateTime.Now:o} WebView2 initialization failed{Environment.NewLine}{ex}{Environment.NewLine}");
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
            File.AppendAllText(Path.Combine(Path.GetTempPath(), "skinsight_desktop_startup.log"),
                $"ERROR {DateTime.Now:o} WebView2 init failed{Environment.NewLine}{e.InitializationException}{Environment.NewLine}");
            MessageBox.Show(
                $"WebView2 init failed: {e.InitializationException?.Message}",
                "SkinSight Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            Close();
            return;
        }

        // Navigate to the embedded server
        _webView.CoreWebView2.Navigate(_serverUrl + "/");
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        base.OnFormClosed(e);
        try
        {
            if (_backendProcess != null && !_backendProcess.HasExited)
                _backendProcess.Kill(entireProcessTree: true);
        }
        catch
        {
            // Best-effort cleanup; the OS will clean up if the child already exited.
        }
    }
}
