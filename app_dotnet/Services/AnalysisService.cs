using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SkinSight.Models;

namespace SkinSight.Services;

/// <summary>
/// Full analysis pipeline: tile extraction → feature extraction (ONNX) → MIL inference (ONNX).
/// 
/// The Python version uses PyTorch (ResNet18 + GatedAttentionMIL) at runtime.
/// This .NET version uses ONNX Runtime – you need to export the models first:
/// 
///   # In Python, export encoder:
///   torch.onnx.export(encoder, dummy_input, "encoder.onnx",
///                     input_names=["input"], output_names=["features"])
///   
///   # Export MIL model:
///   torch.onnx.export(mil_model, dummy_features, "best_model.onnx",
///                     input_names=["features"], output_names=["logits", "attention"])
/// 
/// If ONNX models are not found, the service will still work but return dummy predictions
/// (same behaviour as the Python version when checkpoints are missing).
/// </summary>
public class AnalysisService
{
    private readonly ILogger<AnalysisService> _logger;
    private readonly SkinSightConfig _config;
    private readonly SlideCache _slideCache;
    private readonly string _uploadDir;
    private readonly string _resultsDir;
    private readonly ConcurrentDictionary<string, AnalysisJob> _jobs = new();

    // Lazy-loaded ONNX sessions
    private InferenceSession? _encoderSession;
    private InferenceSession? _milSession;
    private bool _modelsChecked = false;

    // Use the shared 4-class mapping
    private static Dictionary<int, string> ClassNames => ClassInfo.Names;
    private static int NClasses => ClassInfo.Count;

    // ImageNet normalization constants
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    public AnalysisService(
        ILogger<AnalysisService> logger,
        SkinSightConfig config,
        SlideCache slideCache)
    {
        _logger = logger;
        _config = config;
        _slideCache = slideCache;

        var appDir = AppContext.BaseDirectory;
        // Navigate to the project source directory for relative paths
        var projectDir = Path.GetFullPath(Path.Combine(appDir, "..", "..", "..", ".."));
        _uploadDir = Path.Combine(projectDir, "app_dotnet", "uploads");
        _resultsDir = Path.Combine(projectDir, "app_dotnet", "results");

        Directory.CreateDirectory(_uploadDir);
        Directory.CreateDirectory(_resultsDir);
    }

    public string UploadDir => _uploadDir;
    public string ResultsDir => _resultsDir;
    public ConcurrentDictionary<string, AnalysisJob> Jobs => _jobs;

    // ─── Analysis Pipeline ──────────────────────────────────────────

    public void StartAnalysis(string jobId, string slidePath)
    {
        Task.Run(() => RunAnalysis(jobId, slidePath));
    }

    private void RunAnalysis(string jobId, string slidePath)
    {
        try
        {
            // Get model info from the job
            var modelDisplay = "Phikon";
            var modelKey = "phikon";
            if (_jobs.TryGetValue(jobId, out var jobInfo))
            {
                modelDisplay = jobInfo.ModelDisplay;
                modelKey = jobInfo.ModelKey;
            }

            UpdateJob(jobId, status: "processing", progress: 5,
                message: $"Opening slide... (Model: {modelDisplay})");

            using var slide = new OpenSlideInterop.Slide(slidePath);
            var mppStr = slide.GetProperty("openslide.mpp-x");
            double mpp = double.TryParse(mppStr, out var m) ? m : 0.5;

            var slideInfo = new SlideInfo
            {
                Width = slide.Width,
                Height = slide.Height,
                Mpp = Math.Round(mpp, 4),
                Vendor = slide.GetProperty("openslide.vendor") ?? "unknown",
                LevelCount = slide.LevelCount
            };

            UpdateJob(jobId, progress: 10,
                message: "Extracting tiles for analysis...",
                slideInfo: slideInfo);

            // Step 1: Extract tiles
            var (tiles, coords) = ExtractTiles(slide, jobId);
            if (tiles.Count == 0)
            {
                UpdateJob(jobId, status: "error",
                    message: "No tissue tiles found in slide.");
                return;
            }

            UpdateJob(jobId, progress: 40,
                message: $"Extracted {tiles.Count} tiles. Running feature extraction...");

            // Step 2: Feature extraction (ONNX)
            float[,] features = ExtractFeatures(tiles);
            UpdateJob(jobId, progress: 65, message: "Running MIL inference...");

            // Step 3: MIL inference (ONNX)
            var (prediction, probabilities, attentionWeights) = RunMilInference(features);
            UpdateJob(jobId, progress: 80, message: "Generating heatmap...");

            // Step 4: Generate heatmap
            bool heatmapOk = GenerateHeatmap(slide, coords, attentionWeights, jobId);

            // Step 5: Top attention tiles
            var topTiles = GetTopAttentionTiles(tiles, coords, attentionWeights, jobId);

            // Build result
            var probDict = new Dictionary<string, double>();
            for (int i = 0; i < probabilities.Length; i++)
                probDict[ClassNames[i]] = Math.Round(probabilities[i], 4);

            var result = new AnalysisResult
            {
                Prediction = ClassNames[prediction],
                PredictionId = prediction,
                Probabilities = probDict,
                NTiles = tiles.Count,
                TopTiles = topTiles,
                HeatmapAvailable = heatmapOk,
                ModelUsed = modelDisplay,
                ModelKey = modelKey,
                Timestamp = DateTime.Now.ToString("o")
            };

            UpdateJob(jobId, status: "completed", progress: 100,
                message: "Analysis complete!", result: result);

            _logger.LogInformation("Analysis complete for {JobId}: {Prediction}",
                jobId, ClassNames[prediction]);

            // Cleanup tiles in memory
            foreach (var tile in tiles) tile.Dispose();

            if (_config.DeleteSlideAfterAnalysis)
                CleanupSlide(jobId, slidePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Analysis failed for {JobId}", jobId);
            UpdateJob(jobId, status: "error", message: ex.Message);
        }
    }

    // ─── Tile Extraction ────────────────────────────────────────────

    private (List<Image<Rgba32>> tiles, List<TileCoord> coords) ExtractTiles(
        OpenSlideInterop.Slide slide, string jobId)
    {
        var tiles = new List<Image<Rgba32>>();
        var coords = new List<TileCoord>();

        long w = slide.Width, h = slide.Height;
        var mppStr = slide.GetProperty("openslide.mpp-x");
        double mpp = double.TryParse(mppStr, out var m) ? m : 0.5;
        double targetDs = mpp > 0 ? mpp / 0.5 : 1.0;
        int level = slide.GetBestLevelForDownsample(Math.Max(targetDs, 1.0));
        double levelDs = slide.GetLevelDownsample(level);
        int readSize = (int)(_config.TileSize * levelDs);

        // Fast tissue detection via thumbnail
        using var thumb = slide.GetThumbnail(512, 512);
        var thumbBytes = GetRgbBytes(thumb);
        int thumbW = thumb.Width, thumbH = thumb.Height;

        // Simple tissue mask (same logic as Python)
        var tissueMask = new bool[thumbH, thumbW];
        for (int ty = 0; ty < thumbH; ty++)
        {
            for (int tx = 0; tx < thumbW; tx++)
            {
                int idx = (ty * thumbW + tx) * 3;
                float gray = (thumbBytes[idx] + thumbBytes[idx + 1] + thumbBytes[idx + 2]) / 3f;
                tissueMask[ty, tx] = gray < 220 && gray > 30;
            }
        }

        double scaleX = (double)w / thumbW;
        double scaleY = (double)h / thumbH;

        var positions = new List<(int x, int y)>();
        int step = Math.Max(1, thumbH / 50);
        for (int ty = 0; ty < thumbH; ty += step)
        {
            for (int tx = 0; tx < thumbW; tx += step)
            {
                if (tissueMask[ty, tx])
                {
                    int x = (int)(tx * scaleX);
                    int y = (int)(ty * scaleY);
                    if (x + readSize <= w && y + readSize <= h)
                        positions.Add((x, y));
                }
            }
        }

        // Shuffle with fixed seed
        var rng = new Random(42);
        positions = positions.OrderBy(_ => rng.Next()).ToList();
        positions = positions.Take(_config.MaxTilesForAnalysis * 3).ToList();

        var tileSaveDir = Path.Combine(_resultsDir, jobId, "tiles");
        Directory.CreateDirectory(tileSaveDir);

        foreach (var (px, py) in positions)
        {
            if (tiles.Count >= _config.MaxTilesForAnalysis) break;

            var region = slide.ReadRegion(px, py, level, _config.TileSize, _config.TileSize);

            // Check tissue fraction
            var rgbBytes = GetRgbBytes(region);
            int tissuePixels = 0, totalPixels = _config.TileSize * _config.TileSize;
            for (int i = 0; i < totalPixels; i++)
            {
                float gray = (rgbBytes[i * 3] + rgbBytes[i * 3 + 1] + rgbBytes[i * 3 + 2]) / 3f;
                if (gray < 220 && gray > 30) tissuePixels++;
            }
            double tissueFrac = (double)tissuePixels / totalPixels;

            if (tissueFrac < _config.MinTissueFraction)
            {
                region.Dispose();
                continue;
            }

            int idx2 = tiles.Count;
            tiles.Add(region);
            coords.Add(new TileCoord
            {
                X = px, Y = py,
                Level = level,
                Size = _config.TileSize,
                ReadSize = readSize,
                LevelDs = levelDs
            });

            // Save as JPEG
            var savePath = Path.Combine(tileSaveDir, $"tile_{idx2:D4}.jpg");
            region.SaveAsJpeg(savePath);
        }

        _logger.LogInformation("Extracted {Count} tiles from {Candidates} candidates",
            tiles.Count, positions.Count);
        return (tiles, coords);
    }

    // ─── Feature Extraction (ONNX) ─────────────────────────────────

    private float[,] ExtractFeatures(List<Image<Rgba32>> tiles)
    {
        EnsureModels();
        int featureDim = 512; // ResNet18 output
        var allFeatures = new float[tiles.Count, featureDim];

        if (_encoderSession == null)
        {
            // No encoder model → random features (for testing)
            _logger.LogWarning("No ONNX encoder model – generating random features");
            var rng = new Random(42);
            for (int i = 0; i < tiles.Count; i++)
                for (int j = 0; j < featureDim; j++)
                    allFeatures[i, j] = (float)(rng.NextDouble() * 2 - 1);
            return allFeatures;
        }

        // Process in batches
        for (int batchStart = 0; batchStart < tiles.Count; batchStart += 32)
        {
            int batchSize = Math.Min(32, tiles.Count - batchStart);
            var inputTensor = new DenseTensor<float>(new[] { batchSize, 3, 224, 224 });

            for (int b = 0; b < batchSize; b++)
            {
                var tile = tiles[batchStart + b].Clone();
                tile.Mutate(x => x.Resize(224, 224));

                tile.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < 224; y++)
                    {
                        var row = accessor.GetRowSpan(y);
                        for (int x = 0; x < 224; x++)
                        {
                            var pixel = row[x];
                            inputTensor[b, 0, y, x] = (pixel.R / 255f - Mean[0]) / Std[0];
                            inputTensor[b, 1, y, x] = (pixel.G / 255f - Mean[1]) / Std[1];
                            inputTensor[b, 2, y, x] = (pixel.B / 255f - Mean[2]) / Std[2];
                        }
                    }
                });
                tile.Dispose();
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using var results = _encoderSession.Run(inputs);
            var output = results.First().AsTensor<float>();

            for (int b = 0; b < batchSize; b++)
                for (int f = 0; f < featureDim; f++)
                    allFeatures[batchStart + b, f] = output[b, f];
        }

        return allFeatures;
    }

    // ─── MIL Inference (ONNX) ──────────────────────────────────────

    private (int prediction, float[] probabilities, float[] attention)
        RunMilInference(float[,] features)
    {
        int nTiles = features.GetLength(0);
        int featureDim = features.GetLength(1);

        EnsureModels();

        if (_milSession == null)
        {
            _logger.LogWarning("MIL model not available, returning dummy prediction");
            var dummyProbs = new float[] { 0.25f, 0.25f, 0.25f, 0.25f };
            var dummyAttn = new float[nTiles];
            Array.Fill(dummyAttn, 1f / nTiles);
            return (0, dummyProbs, dummyAttn);
        }

        // Create input tensor
        var inputTensor = new DenseTensor<float>(new[] { nTiles, featureDim });
        for (int i = 0; i < nTiles; i++)
            for (int j = 0; j < featureDim; j++)
                inputTensor[i, j] = features[i, j];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("features", inputTensor)
        };

        using var results = _milSession.Run(inputs);
        var resultList = results.ToList();

        // Logits → softmax → probabilities
        var logits = resultList[0].AsTensor<float>();
        var attention = resultList[1].AsTensor<float>();

        var probs = new float[NClasses];
        float maxLogit = float.MinValue;
        for (int c = 0; c < NClasses; c++)
            maxLogit = Math.Max(maxLogit, logits[0, c]);

        float sumExp = 0;
        for (int c = 0; c < NClasses; c++)
        {
            probs[c] = MathF.Exp(logits[0, c] - maxLogit);
            sumExp += probs[c];
        }
        for (int c = 0; c < NClasses; c++)
            probs[c] /= sumExp;

        int pred = Array.IndexOf(probs, probs.Max());

        // Flatten attention weights
        var attn = new float[nTiles];
        for (int i = 0; i < nTiles; i++)
            attn[i] = attention[i, 0];

        return (pred, probs, attn);
    }

    // ─── Heatmap Generation ─────────────────────────────────────────

    private bool GenerateHeatmap(
        OpenSlideInterop.Slide slide,
        List<TileCoord> coords,
        float[] attentionWeights,
        string jobId)
    {
        try
        {
            // Pick a manageable level (< 4096px)
            int useLevel = slide.LevelCount - 1;
            for (int lv = 0; lv < slide.LevelCount; lv++)
            {
                var (lw2, lh2) = slide.GetLevelDimensions(lv);
                if (lw2 <= 4096 && lh2 <= 4096) { useLevel = lv; break; }
            }

            var (wL, hL) = slide.GetLevelDimensions(useLevel);
            double ds = slide.GetLevelDownsample(useLevel);

            using var thumbImg = slide.ReadRegion(0, 0, useLevel, (int)wL, (int)hL);
            var thumbFloat = ImageToFloatArray(thumbImg);

            // Tissue mask
            var tissueMask = new float[hL, wL];
            for (int y = 0; y < hL; y++)
            {
                for (int x = 0; x < wL; x++)
                {
                    float gray = (thumbFloat[y, x, 0] + thumbFloat[y, x, 1] + thumbFloat[y, x, 2]) / 3f;
                    tissueMask[y, x] = (gray < 0.90f && gray > 0.08f) ? 1f : 0f;
                }
            }

            // Build raw attention map
            var heat = new float[hL, wL];
            float[] attn = (float[])attentionWeights.Clone();
            NormalizeMinMax(attn);

            int minStamp = Math.Max(4, (int)(Math.Min(wL, hL) * 0.005));

            for (int i = 0; i < coords.Count && i < attn.Length; i++)
            {
                var coord = coords[i];
                int xL = (int)(coord.X / ds);
                int yL = (int)(coord.Y / ds);
                int stamp = Math.Max(minStamp, (int)(coord.ReadSize / ds));

                int x2 = Math.Min((int)wL, xL + stamp);
                int y2 = Math.Min((int)hL, yL + stamp);
                if (xL < 0 || yL < 0 || xL >= wL || yL >= hL) continue;

                for (int y = yL; y < y2; y++)
                    for (int x = xL; x < x2; x++)
                        heat[y, x] = Math.Max(heat[y, x], attn[i]);
            }

            // Simple box blur (approximation of Gaussian)
            heat = BoxBlur(heat, (int)wL, (int)hL, 5);
            heat = BoxBlur(heat, (int)wL, (int)hL, 11);

            // Normalize
            NormalizeMinMax2D(heat, (int)wL, (int)hL);

            // Power scaling (gentle boost)
            for (int y = 0; y < hL; y++)
                for (int x = 0; x < wL; x++)
                {
                    heat[y, x] = MathF.Pow(heat[y, x], 0.65f);
                    if (heat[y, x] < 0.02f) heat[y, x] = 0f;
                    heat[y, x] *= tissueMask[y, x];
                }

            // Clinical colormap (yellow → orange → red → pink)
            var cmap = MakeClinicalColormap(256);

            // Composite overlay
            var overlayImg = new Image<Rgba32>((int)wL, (int)hL);
            var heatOnlyImg = new Image<Rgba32>((int)wL, (int)hL);

            overlayImg.ProcessPixelRows(thumbImg, (overlayAccessor, thumbAccessor) =>
            {
                for (int y = 0; y < (int)hL; y++)
                {
                    var overlayRow = overlayAccessor.GetRowSpan(y);
                    var thumbRow = thumbAccessor.GetRowSpan(y);
                    for (int x = 0; x < (int)wL; x++)
                    {
                        float h2 = heat[y, x];
                        int ci = Math.Clamp((int)(h2 * 255), 0, 255);
                        float alpha = Math.Clamp(h2 * 0.65f, 0f, 0.65f);

                        float tr = thumbRow[x].R / 255f;
                        float tg = thumbRow[x].G / 255f;
                        float tb = thumbRow[x].B / 255f;

                        float or2 = (1 - alpha) * tr + alpha * cmap[ci, 0];
                        float og = (1 - alpha) * tg + alpha * cmap[ci, 1];
                        float ob = (1 - alpha) * tb + alpha * cmap[ci, 2];

                        overlayRow[x] = new Rgba32(
                            (byte)Math.Clamp(or2 * 255, 0, 255),
                            (byte)Math.Clamp(og * 255, 0, 255),
                            (byte)Math.Clamp(ob * 255, 0, 255),
                            255);
                    }
                }
            });

            heatOnlyImg.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < (int)hL; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < (int)wL; x++)
                    {
                        float h2 = heat[y, x];
                        int ci = Math.Clamp((int)(h2 * 255), 0, 255);
                        byte a = (byte)Math.Clamp(h2 * 200, 0, 200);
                        row[x] = new Rgba32(
                            (byte)(cmap[ci, 0] * 255),
                            (byte)(cmap[ci, 1] * 255),
                            (byte)(cmap[ci, 2] * 255),
                            a);
                    }
                }
            });

            var outDir = Path.Combine(_resultsDir, jobId);
            Directory.CreateDirectory(outDir);

            overlayImg.SaveAsJpeg(Path.Combine(outDir, "heatmap.jpg"));
            thumbImg.SaveAsJpeg(Path.Combine(outDir, "thumbnail.jpg"));
            heatOnlyImg.SaveAsPng(Path.Combine(outDir, "heatmap_only.png"));

            overlayImg.Dispose();
            heatOnlyImg.Dispose();

            _logger.LogInformation("Heatmap saved for {JobId} ({W}x{H})", jobId, wL, hL);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Heatmap generation failed");
            return false;
        }
    }

    // ─── Top Attention Tiles ────────────────────────────────────────

    private List<TopTile> GetTopAttentionTiles(
        List<Image<Rgba32>> tiles,
        List<TileCoord> coords,
        float[] attentionWeights,
        string jobId,
        int topK = 8)
    {
        var indices = attentionWeights
            .Select((val, idx) => (val, idx))
            .OrderByDescending(x => x.val)
            .Take(topK)
            .ToList();

        var topTiles = new List<TopTile>();
        int rank = 1;
        foreach (var (attn, idx) in indices)
        {
            if (idx < tiles.Count && idx < coords.Count)
            {
                topTiles.Add(new TopTile
                {
                    Rank = rank++,
                    TileIndex = idx,
                    Attention = Math.Round(attn, 6),
                    Coord = coords[idx],
                    ImageUrl = $"/api/results/{jobId}/tiles/tile_{idx:D4}.jpg"
                });
            }
        }
        return topTiles;
    }

    // ─── Job Management ─────────────────────────────────────────────

    public void UpdateJob(string jobId,
        string? status = null, int? progress = null,
        string? message = null, SlideInfo? slideInfo = null,
        AnalysisResult? result = null)
    {
        _jobs.AddOrUpdate(jobId,
            _ => new AnalysisJob
            {
                JobId = jobId,
                Status = status ?? "queued",
                Progress = progress ?? 0,
                Message = message ?? "",
                SlideInfo = slideInfo,
                Result = result,
            },
            (_, existing) =>
            {
                if (status != null) existing.Status = status;
                if (progress != null) existing.Progress = progress.Value;
                if (message != null) existing.Message = message;
                if (slideInfo != null) existing.SlideInfo = slideInfo;
                if (result != null) existing.Result = result;
                return existing;
            });
    }

    public void CleanupOldResults()
    {
        if (_config.ResultRetentionMinutes <= 0) return;
        var cutoff = DateTime.Now.AddMinutes(-_config.ResultRetentionMinutes);

        var expired = _jobs.Where(kvp => kvp.Value.CreatedAt < cutoff)
            .Select(kvp => kvp.Key).ToList();

        foreach (var jid in expired)
        {
            var resultDir = Path.Combine(_resultsDir, jid);
            var uploadDir = Path.Combine(_uploadDir, jid);
            foreach (var d in new[] { resultDir, uploadDir })
            {
                if (Directory.Exists(d))
                    Directory.Delete(d, recursive: true);
            }
            _jobs.TryRemove(jid, out _);
            _logger.LogInformation("[cleanup] Expired job {JobId}", jid);
        }
    }

    private void CleanupSlide(string jobId, string slidePath)
    {
        try
        {
            _slideCache.Remove(slidePath);
            if (File.Exists(slidePath)) File.Delete(slidePath);
            var uploadDir = Path.Combine(_uploadDir, jobId);
            if (Directory.Exists(uploadDir) && !Directory.EnumerateFileSystemEntries(uploadDir).Any())
                Directory.Delete(uploadDir);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[cleanup] Could not delete slide for {JobId}", jobId);
        }
    }

    // ─── Model Loading ──────────────────────────────────────────────

    private void EnsureModels()
    {
        if (_modelsChecked) return;
        _modelsChecked = true;

        var appDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";

        var encoderPath = Path.GetFullPath(Path.Combine(appDir, _config.OnnxEncoderPath));
        if (File.Exists(encoderPath))
        {
            _encoderSession = new InferenceSession(encoderPath);
            _logger.LogInformation("ONNX encoder loaded from {Path}", encoderPath);
        }
        else
        {
            _logger.LogWarning("ONNX encoder not found at {Path} – using dummy features", encoderPath);
        }

        var milPath = Path.GetFullPath(Path.Combine(appDir, _config.OnnxModelPath));
        if (File.Exists(milPath))
        {
            _milSession = new InferenceSession(milPath);
            _logger.LogInformation("ONNX MIL model loaded from {Path}", milPath);
        }
        else
        {
            _logger.LogWarning("ONNX MIL model not found at {Path} – using dummy predictions", milPath);
        }
    }

    // ─── Helper Methods ─────────────────────────────────────────────

    private static byte[] GetRgbBytes(Image<Rgba32> image)
    {
        var bytes = new byte[image.Width * image.Height * 3];
        image.ProcessPixelRows(accessor =>
        {
            int i = 0;
            for (int y = 0; y < image.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    bytes[i++] = row[x].R;
                    bytes[i++] = row[x].G;
                    bytes[i++] = row[x].B;
                }
            }
        });
        return bytes;
    }

    private static float[,,] ImageToFloatArray(Image<Rgba32> image)
    {
        var arr = new float[image.Height, image.Width, 3];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < image.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    arr[y, x, 0] = row[x].R / 255f;
                    arr[y, x, 1] = row[x].G / 255f;
                    arr[y, x, 2] = row[x].B / 255f;
                }
            }
        });
        return arr;
    }

    private static void NormalizeMinMax(float[] arr)
    {
        float min = arr.Min(), max = arr.Max();
        if (max - min < 1e-8f) return;
        for (int i = 0; i < arr.Length; i++)
            arr[i] = (arr[i] - min) / (max - min);
    }

    private static void NormalizeMinMax2D(float[,] arr, int w, int h)
    {
        float min = float.MaxValue, max = float.MinValue;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                min = Math.Min(min, arr[y, x]);
                max = Math.Max(max, arr[y, x]);
            }
        if (max - min < 1e-8f) return;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                arr[y, x] = (arr[y, x] - min) / (max - min);
    }

    private static float[,] BoxBlur(float[,] src, int w, int h, int radius)
    {
        var dst = new float[h, w];
        int r = radius / 2;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float sum = 0;
                int count = 0;
                for (int dy = -r; dy <= r; dy++)
                {
                    for (int dx = -r; dx <= r; dx++)
                    {
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < h && nx >= 0 && nx < w)
                        {
                            sum += src[ny, nx];
                            count++;
                        }
                    }
                }
                dst[y, x] = sum / count;
            }
        }
        return dst;
    }

    private static float[,] MakeClinicalColormap(int n = 256)
    {
        var cmap = new float[n, 3];
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / (n - 1);
            if (t < 0.25f)
            {
                float s = t / 0.25f;
                cmap[i, 0] = 1.0f;
                cmap[i, 1] = 0.95f - 0.15f * s;
                cmap[i, 2] = 0.2f * (1 - s);
            }
            else if (t < 0.5f)
            {
                float s = (t - 0.25f) / 0.25f;
                cmap[i, 0] = 1.0f;
                cmap[i, 1] = 0.8f - 0.35f * s;
                cmap[i, 2] = 0.0f;
            }
            else if (t < 0.75f)
            {
                float s = (t - 0.5f) / 0.25f;
                cmap[i, 0] = 1.0f;
                cmap[i, 1] = 0.45f - 0.4f * s;
                cmap[i, 2] = 0.05f * s;
            }
            else
            {
                float s = (t - 0.75f) / 0.25f;
                cmap[i, 0] = 1.0f;
                cmap[i, 1] = 0.05f;
                cmap[i, 2] = 0.05f + 0.45f * s;
            }
        }
        return cmap;
    }
}
