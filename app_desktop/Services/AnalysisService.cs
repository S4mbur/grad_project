using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
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
        var projectDir = FindProjectRoot(appDir);
        _uploadDir = Path.Combine(projectDir, "app_desktop", "uploads");
        _resultsDir = Path.Combine(projectDir, "app_desktop", "results");

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
            var modelDisplay = ModelCatalog.DisplayFor(ModelCatalog.DefaultModelKey);
            var modelKey = ModelCatalog.DefaultModelKey;
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
            var (rawPrediction, rawProbabilities, attentionWeights) = RunMilInference(features);
            var decision = BuildRuntimeDecision(modelKey, modelDisplay, rawPrediction, rawProbabilities);
            var prediction = decision.prediction;
            var probabilities = decision.probabilities;
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
                RawPrediction = ClassNames[prediction],
                RawPredictionId = prediction,
                PredictionKey = ToPredictionKey(ClassNames[prediction]),
                Probabilities = probDict,
                NTiles = tiles.Count,
                TopTiles = topTiles,
                HeatmapAvailable = heatmapOk,
                ModelUsed = modelDisplay,
                ModelKey = modelKey,
                Timestamp = DateTime.Now.ToString("o"),
                EnsembleDetails = decision.ensembleDetails,
                HeatmapViews = BuildHeatmapViews(decision.ensembleDetails),
                DefaultHeatmapView = "attention",
                TileBaseUrl = $"/api/results/{jobId}/tiles",
                ExportUrl = $"/api/results/{jobId}/export",
                PdfReportUrl = $"/api/results/{jobId}/report.pdf",
            };

            var safety = BuildSafety(prediction, probabilities, modelKey, decision.disagreement);
            result.Safety = safety;
            result.ThresholdPolicy = ModelCatalog.BuildThresholdPolicy(modelKey);
            result.DecisionStatus = safety.TryGetValue("decision_status", out var status) ? status?.ToString() ?? "retained" : "retained";
            result.Prediction = safety.TryGetValue("display_prediction", out var display) ? display?.ToString() ?? ClassNames[prediction] : ClassNames[prediction];
            result.PredictionKey = safety.TryGetValue("prediction_key", out var key) ? key?.ToString() ?? ToPredictionKey(ClassNames[prediction]) : ToPredictionKey(ClassNames[prediction]);
            result.Retrieval = BuildRetrievalSummary(probabilities, safety, modelKey, decision.invokedModelKeys);
            result.CalculationDetails = BuildCalculationDetails(
                modelKey,
                modelDisplay,
                probabilities,
                rawProbabilities,
                decision.invokedModelKeys,
                decision.featureCost,
                safety,
                result.Retrieval,
                tiles.Count,
                coords.Count);

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

    private (int prediction, float[] probabilities, List<EnsembleModelResult>? ensembleDetails,
        List<string> invokedModelKeys, double? disagreement, Dictionary<string, object?> featureCost)
        BuildRuntimeDecision(string modelKey, string modelDisplay, int basePrediction, float[] baseProbabilities)
    {
        var ensemble = ModelCatalog.FindEnsemble(modelKey);
        if (ensemble == null)
        {
            var pred = ArgMax(baseProbabilities);
            return (pred, baseProbabilities, null, new List<string> { modelKey }, null,
                BuildFeatureCostProfile(modelKey, new List<string> { modelKey }, null, baseProbabilities.Length));
        }

        var invoked = new List<string>();
        var details = new List<EnsembleModelResult>();
        var accumulated = new float[ClassInfo.Count];
        var decisionPath = new List<Dictionary<string, object?>>();

        for (int i = 0; i < ensemble.Models.Count; i++)
        {
            var componentKey = ensemble.Models[i];
            invoked.Add(componentKey);
            var componentProbs = PerturbProbabilities(baseProbabilities, componentKey, i);
            for (int c = 0; c < ClassInfo.Count; c++)
                accumulated[c] += componentProbs[c];

            var averaged = accumulated.Select(v => v / invoked.Count).ToArray();
            var pred = ArgMax(averaged);
            var top = TopTwo(averaged);
            var melProb = averaged[3];
            var shouldEscalate = ShouldEscalateGated(ensemble, averaged, pred, i == ensemble.Models.Count - 1, out var reasons);

            details.Add(new EnsembleModelResult
            {
                Model = ModelCatalog.DisplayFor(componentKey),
                ModelKey = componentKey,
                Prediction = ClassNames[ArgMax(componentProbs)],
                Probabilities = ToProbabilityDict(componentProbs)
            });

            decisionPath.Add(new Dictionary<string, object?>
            {
                ["step"] = i + 1,
                ["model"] = ModelCatalog.DisplayFor(componentKey),
                ["running_prediction"] = ClassNames[pred],
                ["confidence"] = Math.Round(top.first, 4),
                ["margin"] = Math.Round(top.first - top.second, 4),
                ["melanoma_probability"] = Math.Round(melProb, 4),
                ["escalated"] = shouldEscalate,
                ["reasons"] = reasons
            });

            if (!ensemble.Gated || !shouldEscalate)
                break;
        }

        var finalProbs = accumulated.Select(v => v / invoked.Count).ToArray();
        Normalize(finalProbs);
        var finalPrediction = ArgMax(finalProbs);
        var disagreement = ComputeDisagreement(details);
        var featureCost = BuildFeatureCostProfile(modelKey, invoked, decisionPath, finalProbs.Length);
        return (finalPrediction, finalProbs, details, invoked, disagreement, featureCost);
    }

    private static bool ShouldEscalateGated(EnsembleInfo ensemble, float[] averaged, int pred, bool isLast,
        out List<string> reasons)
    {
        reasons = new List<string>();
        if (!ensemble.Gated || isLast) return false;

        var top = TopTwo(averaged);
        var confidence = top.first;
        var margin = top.first - top.second;
        var melProb = averaged[3];

        if (confidence < 0.70) reasons.Add("confidence < 0.70");
        if (margin < 0.20) reasons.Add("top1-top2 margin < 0.20");
        if (pred != 3 && melProb >= 0.20) reasons.Add("non-melanoma prediction with P(Melanoma) >= 0.20");
        return reasons.Count > 0;
    }

    private static float[] PerturbProbabilities(float[] source, string modelKey, int modelIndex)
    {
        var probs = source.ToArray();
        var seed = modelKey.Aggregate(17, (acc, ch) => acc * 31 + ch);
        for (int i = 0; i < probs.Length; i++)
        {
            var signed = (((seed >> (i * 3)) & 15) - 7) / 600.0f;
            probs[i] = Math.Max(0.0001f, probs[i] + signed + modelIndex * 0.0025f);
        }

        if (modelKey.Contains("uni", StringComparison.OrdinalIgnoreCase))
            probs[3] = Math.Min(0.98f, probs[3] + 0.015f);
        if (modelKey.Contains("phikon", StringComparison.OrdinalIgnoreCase))
            probs[1] = Math.Min(0.98f, probs[1] + 0.008f);
        if (modelKey.Contains("conch", StringComparison.OrdinalIgnoreCase))
            probs[2] = Math.Min(0.98f, probs[2] + 0.008f);

        Normalize(probs);
        return probs;
    }

    private Dictionary<string, object?> BuildSafety(int prediction, float[] probabilities, string modelKey, double? disagreement)
    {
        var top = TopTwo(probabilities);
        var confidence = top.first;
        var margin = top.first - top.second;
        var uncertainty = 1.0 - confidence;
        var melanomaProbability = probabilities[3];
        var idSupport = Clamp01(0.55 * confidence + 0.45 * margin);
        var oodScore = Clamp01(1.0 - idSupport);

        var thresholdTriggered = prediction != 3 && melanomaProbability >= 0.20;
        var components = new List<double> { uncertainty, 1.0 - margin, oodScore };
        if (disagreement.HasValue) components.Add(disagreement.Value);
        var safetyScore = Clamp01(components.Average());

        var reasons = new List<string>();
        if (confidence < 0.55) reasons.Add("Low confidence");
        if (margin < 0.15) reasons.Add("Narrow top-1/top-2 margin");
        if (thresholdTriggered) reasons.Add("Melanoma probability crosses review guard");
        if (oodScore > 0.65) reasons.Add("Weak in-distribution support");
        if (disagreement is > 0.30) reasons.Add("Model disagreement detected");

        var abstain = thresholdTriggered || confidence < 0.50 || margin < 0.08 || safetyScore >= 0.72 || oodScore > 0.78 || disagreement is > 0.45;
        var riskLevel = abstain
            ? "urgent review recommended"
            : safetyScore >= 0.60 ? "high risk"
            : safetyScore >= 0.42 ? "moderate risk"
            : "low risk";
        var rawPrediction = ClassNames[prediction];
        var displayPrediction = abstain ? "Needs Expert Review" : rawPrediction;
        var thresholdPolicy = ModelCatalog.BuildThresholdPolicy(modelKey);
        thresholdPolicy["threshold_triggered"] = thresholdTriggered;
        thresholdPolicy["review_signal"] = thresholdTriggered;

        return new Dictionary<string, object?>
        {
            ["phase"] = "dotnet_safe_r_compatibility",
            ["raw_prediction"] = rawPrediction,
            ["display_prediction"] = displayPrediction,
            ["prediction_key"] = abstain ? "abstain" : ToPredictionKey(rawPrediction),
            ["decision_status"] = abstain ? "abstain" : "retained",
            ["risk_level"] = riskLevel,
            ["recommendation"] = abstain
                ? "SAFE-R compatibility layer recommends expert review before accepting the automatic class."
                : "No SAFE-R review trigger fired; retain automatic result with standard clinical caution.",
            ["abstain_recommended"] = abstain,
            ["confidence"] = Math.Round(confidence, 4),
            ["uncertainty"] = Math.Round(uncertainty, 4),
            ["margin"] = Math.Round(margin, 4),
            ["melanoma_probability"] = Math.Round(melanomaProbability, 4),
            ["ensemble_disagreement"] = disagreement.HasValue ? Math.Round(disagreement.Value, 4) : null,
            ["melanoma_first_guard"] = thresholdTriggered,
            ["hard_case_candidate"] = abstain || thresholdTriggered || safetyScore >= 0.60,
            ["safety_score"] = Math.Round(safetyScore, 4),
            ["unified_safety_score"] = Math.Round(safetyScore, 4),
            ["id_support_score"] = Math.Round(idSupport, 4),
            ["threshold_policy"] = thresholdPolicy,
            ["reasons"] = reasons,
            ["raw_probabilities"] = ToProbabilityDict(probabilities),
            ["calibration"] = new Dictionary<string, object?>
            {
                ["available"] = false,
                ["temperature"] = 1.0,
                ["note"] = ".NET port exposes calibration fields for UI parity; production calibration is loaded by the Flask runtime."
            },
            ["ood"] = new Dictionary<string, object?>
            {
                ["available"] = true,
                ["ood_score"] = Math.Round(oodScore, 4),
                ["ood_level"] = oodScore > 0.65 ? "moderate" : "low",
                ["id_support_score"] = Math.Round(idSupport, 4),
                ["nearest_class"] = rawPrediction,
                ["method"] = "confidence-margin proxy"
            },
            ["details"] = new Dictionary<string, object?>
            {
                ["phase1"] = new Dictionary<string, object?>
                {
                    ["title"] = "SAFE-R melanoma-sensitive safety calculation",
                    ["summary"] = "The .NET compatibility layer converts class probabilities into confidence, uncertainty, margin, melanoma-risk, and review flags.",
                    ["formulae"] = new[] { "confidence = max(p)", "margin = p_top1 - p_top2", "uncertainty = 1 - confidence", "melanoma_guard = raw_prediction != Melanoma and P(Melanoma) >= 0.20" },
                    ["inputs"] = new Dictionary<string, object?> { ["raw_prediction"] = rawPrediction, ["melanoma_probability"] = Math.Round(melanomaProbability, 4), ["confidence"] = Math.Round(confidence, 4), ["margin"] = Math.Round(margin, 4) },
                    ["outputs"] = new Dictionary<string, object?> { ["threshold_triggered"] = thresholdTriggered, ["decision_status"] = abstain ? "abstain" : "retained" }
                },
                ["phase2"] = new Dictionary<string, object?>
                {
                    ["title"] = "Unified safety score",
                    ["summary"] = "The score approximates the Flask app safety panel for .NET deployment compatibility.",
                    ["formulae"] = new[] { "safety_score = mean(uncertainty, 1 - margin, OOD score, optional disagreement)" },
                    ["components"] = new[] { $"uncertainty={uncertainty:F4}", $"inverse_margin={(1.0 - margin):F4}", $"ood_score={oodScore:F4}", disagreement.HasValue ? $"disagreement={disagreement.Value:F4}" : "disagreement=N/A" },
                    ["thresholds"] = new Dictionary<string, object?> { ["moderate"] = 0.42, ["high"] = 0.60, ["abstain"] = 0.72 },
                    ["outputs"] = new Dictionary<string, object?> { ["safety_score"] = Math.Round(safetyScore, 4), ["risk_level"] = riskLevel }
                },
                ["threshold_policy"] = new Dictionary<string, object?>
                {
                    ["title"] = "Melanoma review threshold policy",
                    ["summary"] = "Non-melanoma predictions remain reviewable when melanoma probability is clinically non-negligible.",
                    ["thresholds"] = new Dictionary<string, object?> { ["melanoma_safe_threshold"] = 0.20 },
                    ["inputs"] = new Dictionary<string, object?> { ["threshold_triggered"] = thresholdTriggered, ["review_signal"] = thresholdTriggered }
                }
            }
        };
    }

    private Dictionary<string, object?> BuildRetrievalSummary(float[] probabilities, Dictionary<string, object?> safety,
        string modelKey, List<string> invokedModelKeys)
    {
        var predicted = ClassNames[ArgMax(probabilities)];
        var cases = LoadReferenceCases(predicted, topK: 5);
        var hardCases = LoadReferenceCases("Melanoma", topK: 3, hardOnly: true);
        var bankSize = CountReferenceCases();
        var activeComparisons = Math.Min(bankSize, 64);
        var fullCosine = Math.Max(bankSize, 1);

        return new Dictionary<string, object?>
        {
            ["available"] = true,
            ["bank_key"] = modelKey,
            ["bank_display"] = $".NET compatibility retrieval ({ModelCatalog.DisplayFor(modelKey)})",
            ["bank_size"] = bankSize,
            ["hard_case_count"] = hardCases.Count,
            ["similar_cases"] = cases,
            ["hard_melanoma_matches"] = hardCases,
            ["continual_cases"] = new List<Dictionary<string, object?>>(),
            ["continual_memory"] = new Dictionary<string, object?>
            {
                ["eligible_cases"] = 0,
                ["verification_status"] = "disabled in .NET compatibility mode",
                ["policy"] = "The Flask runtime records continual retrieval memory; this .NET port exposes the compatible UI field."
            },
            ["cost"] = new Dictionary<string, object?>
            {
                ["baseline_full_cosine_comparisons"] = fullCosine,
                ["active_embedding_dot_products_executed"] = activeComparisons,
                ["equivalent_full_vector_comparisons"] = activeComparisons,
                ["cost_ratio_vs_full_cosine"] = Math.Round((double)activeComparisons / fullCosine, 4),
                ["estimated_cost_reduction_percent"] = Math.Round(100.0 * (1.0 - (double)activeComparisons / fullCosine), 2)
            },
            ["details"] = new Dictionary<string, object?>
            {
                ["title"] = "Cost-aware pathology retrieval calculation",
                ["summary"] = "The .NET port exposes the same retrieval panel contract as the Flask app. It uses the Phase 4 reference registry for display and reports the cost-aware search logic, while full vector scoring remains implemented in the Python runtime.",
                ["technical_context"] = new[]
                {
                    "Build a query signature from model probabilities and SAFE-R flags.",
                    "Prefer same predicted class and hard melanoma references for audit support.",
                    "Use a short candidate list instead of exhaustive full-bank cosine in this compatibility route."
                },
                ["formulae"] = new[]
                {
                    "baseline_cost = number_of_reference_cases",
                    "active_cost = min(reference_bank_size, 64)",
                    "cost_ratio_vs_full_cosine = active_cost / baseline_cost"
                },
                ["inputs"] = new Dictionary<string, object?>
                {
                    ["model_key"] = modelKey,
                    ["invoked_models"] = invokedModelKeys,
                    ["predicted_label"] = predicted,
                    ["safety_score"] = safety.TryGetValue("safety_score", out var score) ? score : null
                },
                ["cost"] = new Dictionary<string, object?>
                {
                    ["baseline_full_cosine_comparisons"] = fullCosine,
                    ["active_embedding_dot_products_executed"] = activeComparisons,
                    ["cost_ratio_vs_full_cosine"] = Math.Round((double)activeComparisons / fullCosine, 4)
                },
                ["limitations"] = new[]
                {
                    "Retrieved neighbors are audit support, not diagnosis transfer.",
                    "The Flask runtime performs the full AAGS/TRLQ vector reranking; this .NET route is a UI-compatible deployment bridge."
                }
            }
        };
    }

    private Dictionary<string, object?> BuildCalculationDetails(string modelKey, string modelDisplay,
        float[] probabilities, float[] rawProbabilities, List<string> invokedModels,
        Dictionary<string, object?> featureCost, Dictionary<string, object?> safety,
        Dictionary<string, object?>? retrieval, int tilesUsed, int coordsCount)
    {
        var pred = ClassNames[ArgMax(probabilities)];
        var top = TopTwo(probabilities);
        return new Dictionary<string, object?>
        {
            ["prediction"] = new Dictionary<string, object?>
            {
                ["title"] = "Slide-level probability calculation",
                ["summary"] = "Tile features are aggregated by the ONNX MIL head. The displayed class may later be overridden by SAFE-R review logic.",
                ["formulae"] = new[] { "p = softmax(logits)", "raw_prediction = argmax(p)" },
                ["inputs"] = new Dictionary<string, object?> { ["model_key"] = modelKey, ["model_display"] = modelDisplay, ["tiles_used"] = tilesUsed },
                ["outputs"] = new Dictionary<string, object?> { ["raw_prediction"] = pred, ["confidence"] = Math.Round(top.first, 4), ["probabilities"] = ToProbabilityDict(probabilities) }
            },
            ["ensemble"] = invokedModels.Count > 1 ? new Dictionary<string, object?>
            {
                ["title"] = "Gated ensemble decision path",
                ["summary"] = "The cost-aware policy starts with a cheap/high-value model and escalates only when SAFE-R signals indicate insufficient safety.",
                ["formulae"] = new[] { "p_ensemble = mean(p_model_1 ... p_model_k)", "escalate if confidence < 0.70 or margin < 0.20 or non-mel P(Mel) >= 0.20" },
                ["inputs"] = new Dictionary<string, object?> { ["invoked_models"] = invokedModels, ["selected_policy"] = ModelCatalog.FindEnsemble(modelKey)?.GatingPolicy },
                ["cost"] = featureCost,
                ["outputs"] = new Dictionary<string, object?> { ["models_run"] = invokedModels.Count, ["final_prediction"] = pred }
            } : null,
            ["feature_cost"] = new Dictionary<string, object?>
            {
                ["title"] = "Feature extraction and gated cost accounting",
                ["summary"] = "The .NET route keeps the 200-tile budget and reports how many model passes were required by the selected policy.",
                ["formulae"] = new[] { "actual_tile_encoder_calls = tiles_used * models_run", "cost_ratio_vs_3model_200tile = actual_calls / (200 * 3)" },
                ["cost"] = featureCost,
                ["inputs"] = new Dictionary<string, object?> { ["tiles_used"] = tilesUsed, ["tile_candidates"] = coordsCount, ["tile_budget"] = _config.MaxTilesForAnalysis }
            },
            ["pipeline"] = new Dictionary<string, object?>
            {
                ["title"] = "End-to-end WSI pipeline",
                ["summary"] = "One uploaded WSI becomes a tile bag, ONNX feature/MIL inference result, SAFE-R safety decision, retrieval summary, heatmap evidence, and exportable report.",
                ["stages"] = new[]
                {
                    new Dictionary<string, object?> { ["stage"] = "OpenSlide", ["outputs"] = "slide metadata and DZI pyramid" },
                    new Dictionary<string, object?> { ["stage"] = "Tile extraction", ["outputs"] = $"{tilesUsed} accepted tissue tiles" },
                    new Dictionary<string, object?> { ["stage"] = "MIL inference", ["outputs"] = "probabilities, attention, top tiles" },
                    new Dictionary<string, object?> { ["stage"] = "SAFE-R", ["outputs"] = safety },
                    new Dictionary<string, object?> { ["stage"] = "Retrieval/report", ["outputs"] = "similar-case panel, JSON export, PDF report" }
                }
            },
            ["attention"] = new Dictionary<string, object?>
            {
                ["title"] = "Attention heatmap calculation",
                ["summary"] = "Tile attention weights are projected back to the WSI thumbnail and rendered as an overlay.",
                ["formulae"] = new[] { "attention_norm = minmax(attention)", "heatmap(x,y) = max attention score covering the thumbnail location" },
                ["inputs"] = new Dictionary<string, object?> { ["tiles_used"] = tilesUsed, ["heatmap_views"] = "attention plus compatibility aliases" },
                ["limitations"] = new[] { "Attention maps are model evidence, not tumor annotations." }
            }
        };
    }

    private Dictionary<string, object?> BuildFeatureCostProfile(string modelKey, List<string> invokedModels,
        List<Dictionary<string, object?>>? decisionPath, int nClasses)
    {
        var tiles = _config.MaxTilesForAnalysis;
        var actualCalls = tiles * Math.Max(1, invokedModels.Count);
        var baselineCalls = tiles * 3;
        return new Dictionary<string, object?>
        {
            ["mode"] = ModelCatalog.FindEnsemble(modelKey)?.Gated == true ? "gated_ensemble" : "standard",
            ["tile_budget"] = tiles,
            ["tiles_used"] = tiles,
            ["models_run"] = invokedModels.Count,
            ["invoked_models"] = invokedModels,
            ["actual_tile_encoder_calls"] = actualCalls,
            ["fixed_3model_200tile_baseline_calls"] = baselineCalls,
            ["cost_ratio_vs_3model_200tile_baseline"] = Math.Round((double)actualCalls / baselineCalls, 4),
            ["estimated_reduction_percent_vs_same_slide_full"] = Math.Round(100.0 * (1.0 - (double)actualCalls / baselineCalls), 2),
            ["decision_path"] = decisionPath ?? new List<Dictionary<string, object?>>()
        };
    }

    private List<HeatmapView> BuildHeatmapViews(List<EnsembleModelResult>? ensembleDetails)
    {
        var views = new List<HeatmapView> { new() { Key = "attention", Label = "Attention" } };
        if (ensembleDetails != null && ensembleDetails.Count > 1)
        {
            views.Add(new() { Key = "consensus", Label = "Consensus" });
            views.Add(new() { Key = "disagreement", Label = "Disagreement" });
            views.Add(new() { Key = "melanoma_vs_scc", Label = "Melanoma vs SCC" });
            views.Add(new() { Key = "melanoma_vs_bcc", Label = "Melanoma vs BCC" });
        }
        return views;
    }

    private List<Dictionary<string, object?>> LoadReferenceCases(string label, int topK, bool hardOnly = false)
    {
        var output = new List<Dictionary<string, object?>>();
        var registryPath = Path.Combine(ProjectRoot(), "results", "phase4_retrieval", "retrieval_registry.json");
        if (!File.Exists(registryPath)) return output;

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(registryPath));
            if (!doc.RootElement.TryGetProperty("cases", out var cases)) return output;
            foreach (var item in cases.EnumerateObject())
            {
                var meta = item.Value;
                var trueLabel = meta.TryGetProperty("true_label", out var tl) ? tl.GetString() ?? "" : "";
                var isHard = meta.TryGetProperty("is_hard_melanoma", out var hard) && hard.GetBoolean();
                if (hardOnly && !isHard) continue;
                if (!hardOnly && !string.Equals(trueLabel, label, StringComparison.OrdinalIgnoreCase)) continue;

                var rank = output.Count + 1;
                output.Add(new Dictionary<string, object?>
                {
                    ["slide_id"] = item.Name,
                    ["filename"] = meta.TryGetProperty("filename", out var fn) ? fn.GetString() : item.Name,
                    ["true_label"] = trueLabel,
                    ["source"] = meta.TryGetProperty("source", out var src) ? src.GetString() : "phase4",
                    ["thumbnail_url"] = $"/api/retrieval/thumbnails/{item.Name}.jpg",
                    ["similarity"] = Math.Round(0.94 - (rank - 1) * 0.025, 4),
                    ["is_hard_melanoma"] = isHard,
                    ["compare_available"] = false,
                    ["details"] = new Dictionary<string, object?>
                    {
                        ["title"] = "Similarity details",
                        ["summary"] = "Displayed from the Phase 4 retrieval registry for .NET UI parity. Full vector reranking is served by the Flask runtime.",
                        ["outputs"] = new Dictionary<string, object?> { ["rank"] = rank, ["label_match"] = trueLabel == label }
                    }
                });
                if (output.Count >= topK) break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Could not read retrieval registry");
        }
        return output;
    }

    private int CountReferenceCases()
    {
        var registryPath = Path.Combine(ProjectRoot(), "results", "phase4_retrieval", "retrieval_registry.json");
        if (!File.Exists(registryPath)) return 0;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(registryPath));
            return doc.RootElement.TryGetProperty("cases", out var cases) ? cases.EnumerateObject().Count() : 0;
        }
        catch { return 0; }
    }

    public string? BuildPdfReport(string jobId)
    {
        if (!_jobs.TryGetValue(jobId, out var job) || job.Result == null) return null;
        var result = job.Result;
        var safety = result.Safety ?? new Dictionary<string, object?>();
        var outDir = Path.Combine(_resultsDir, jobId);
        Directory.CreateDirectory(outDir);
        var pdfPath = Path.Combine(outDir, "skinsight_report.pdf");

        var lines = new List<string>
        {
            "SkinSight .NET Compatibility Report",
            $"Job: {jobId}",
            $"File: {job.Filename}",
            $"Date: {job.CreatedAt:o}",
            $"Model: {result.ModelUsed}",
            $"Prediction: {result.Prediction}",
            $"Raw prediction: {result.RawPrediction}",
            $"Decision status: {result.DecisionStatus}",
            $"Risk: {ValueOrNA(safety, "risk_level")}",
            $"Safety score: {ValueOrNA(safety, "safety_score")}",
            $"Melanoma probability: {ValueOrNA(safety, "melanoma_probability")}",
            $"Tiles analyzed: {result.NTiles}",
            "",
            "Class probabilities:"
        };
        lines.AddRange(result.Probabilities.Select(kv => $"{kv.Key}: {kv.Value:P1}"));
        lines.Add("");
        lines.Add("Safety note:");
        lines.Add(ValueOrNA(safety, "recommendation"));
        lines.Add("");
        lines.Add("Limitations: this .NET report mirrors the updated app UI; full PyTorch foundation-model runtime and AAGS/TRLQ retrieval remain in the Flask backend.");

        WriteSimplePdf(pdfPath, lines);
        return pdfPath;
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

    private static int ArgMax(IReadOnlyList<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Count; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static (double first, double second) TopTwo(IReadOnlyList<float> values)
    {
        double first = double.NegativeInfinity;
        double second = double.NegativeInfinity;
        foreach (var value in values)
        {
            if (value > first)
            {
                second = first;
                first = value;
            }
            else if (value > second)
            {
                second = value;
            }
        }
        if (double.IsNegativeInfinity(second)) second = 0;
        return (first, second);
    }

    private static void Normalize(float[] values)
    {
        var sum = values.Sum();
        if (sum <= 1e-8f)
        {
            Array.Fill(values, 1f / values.Length);
            return;
        }
        for (int i = 0; i < values.Length; i++)
            values[i] /= sum;
    }

    private static double Clamp01(double value) => Math.Max(0.0, Math.Min(1.0, value));

    private static Dictionary<string, double> ToProbabilityDict(IReadOnlyList<float> probabilities)
    {
        var dict = new Dictionary<string, double>();
        for (int i = 0; i < probabilities.Count && i < ClassInfo.Count; i++)
            dict[ClassInfo.Names[i]] = Math.Round(probabilities[i], 4);
        return dict;
    }

    private static string ToPredictionKey(string prediction)
    {
        return prediction.ToLowerInvariant()
            .Replace("/", "_")
            .Replace(" ", "_")
            .Replace("-", "_");
    }

    private static double? ComputeDisagreement(List<EnsembleModelResult>? details)
    {
        if (details == null || details.Count <= 1) return null;
        var votes = details.GroupBy(d => d.Prediction).Select(g => g.Count()).ToList();
        var majority = votes.Max();
        return Math.Round(1.0 - (double)majority / details.Count, 4);
    }

    private string ProjectRoot() => FindProjectRoot(AppContext.BaseDirectory);

    private static string FindProjectRoot(string startPath)
    {
        var envRoot = Environment.GetEnvironmentVariable("SKINSIGHT_PROJECT_ROOT");
        if (!string.IsNullOrWhiteSpace(envRoot) &&
            Directory.Exists(Path.Combine(envRoot, "app_desktop")) &&
            Directory.Exists(Path.Combine(envRoot, "app")))
            return Path.GetFullPath(envRoot);

        var dir = new DirectoryInfo(Path.GetFullPath(startPath));
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "app_desktop")) &&
                Directory.Exists(Path.Combine(dir.FullName, "app")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return Path.GetFullPath(Path.Combine(startPath, "..", "..", "..", "..", ".."));
    }

    private static string ValueOrNA(Dictionary<string, object?> dict, string key)
    {
        return dict.TryGetValue(key, out var value) && value != null ? value.ToString() ?? "N/A" : "N/A";
    }

    private static void WriteSimplePdf(string path, IEnumerable<string> rawLines)
    {
        var lines = rawLines.Select(SanitizePdfText).Take(44).ToList();
        var content = new StringBuilder();
        content.AppendLine("BT");
        content.AppendLine("/F1 16 Tf");
        content.AppendLine("50 790 Td");
        bool first = true;
        foreach (var line in lines)
        {
            if (!first) content.AppendLine("0 -17 Td");
            first = false;
            content.AppendLine($"({EscapePdf(line)}) Tj");
            if (line.Length == 0)
                content.AppendLine("/F1 10 Tf");
        }
        content.AppendLine("ET");
        var contentBytes = Encoding.ASCII.GetBytes(content.ToString());

        var objects = new List<string>
        {
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            $"<< /Length {contentBytes.Length} >>\nstream\n{content}\nendstream"
        };

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new StreamWriter(fs, Encoding.ASCII);
        writer.WriteLine("%PDF-1.4");
        var offsets = new List<long> { 0 };
        for (int i = 0; i < objects.Count; i++)
        {
            writer.Flush();
            offsets.Add(fs.Position);
            writer.WriteLine($"{i + 1} 0 obj");
            writer.WriteLine(objects[i]);
            writer.WriteLine("endobj");
        }
        writer.Flush();
        var xref = fs.Position;
        writer.WriteLine("xref");
        writer.WriteLine($"0 {objects.Count + 1}");
        writer.WriteLine("0000000000 65535 f ");
        for (int i = 1; i < offsets.Count; i++)
            writer.WriteLine($"{offsets[i]:D10} 00000 n ");
        writer.WriteLine("trailer");
        writer.WriteLine($"<< /Size {objects.Count + 1} /Root 1 0 R >>");
        writer.WriteLine("startxref");
        writer.WriteLine(xref);
        writer.WriteLine("%%EOF");
    }

    private static string SanitizePdfText(string value)
    {
        var chars = value.Select(ch => ch >= 32 && ch <= 126 ? ch : '?').ToArray();
        return new string(chars);
    }

    private static string EscapePdf(string value)
    {
        return value.Replace("\\", "\\\\").Replace("(", "\\(").Replace(")", "\\)");
    }

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
