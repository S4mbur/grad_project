namespace SkinSight.Models;

/// <summary>Configuration matching the Python app-level runtime switches.</summary>
public class SkinSightConfig
{
    public bool DeleteSlideAfterAnalysis { get; set; } = false;
    public int ResultRetentionMinutes { get; set; } = 60;
    public int MaxUploadSizeGB { get; set; } = 5;
    public int MaxTilesForAnalysis { get; set; } = 200;
    public int TileSize { get; set; } = 256;
    public double MinTissueFraction { get; set; } = 0.3;
    public int DziTileSize { get; set; } = 254;
    public int DziOverlap { get; set; } = 1;
    public int DziQuality { get; set; } = 75;
    public int SlideCacheSize { get; set; } = 4;
    public string OnnxModelPath { get; set; } = "../results/deployment/best_model.onnx";
    public string OnnxEncoderPath { get; set; } = "../results/deployment/encoder.onnx";
}

public static class ClassInfo
{
    public static readonly Dictionary<int, string> Names = new()
    {
        { 0, "Normal/Benign" }, { 1, "BCC" }, { 2, "SCC" }, { 3, "Melanoma" }
    };

    public static readonly Dictionary<string, int> Ids = Names.ToDictionary(kv => kv.Value, kv => kv.Key);
    public static readonly int Count = 4;
}

public static class ModelCatalog
{
    public const string DefaultModelKey = "gated_app_order_cheap_conf70_margin20_mel20";

    public static readonly List<ModelInfo> Models = new()
    {
        new() { Key = "uni_cost_sensitive_strong", Name = "UNI", Group = "UNI", Display = "UNI - Cost-Sensitive Strong", F1 = 0.9541, Auc = 0.9957, MelFn = 1, Description = "UNI encoder with strong cost-sensitive melanoma penalty. Current best overall run: Macro F1 95.4%, AUC 99.6%, Melanoma FN=1.", Available = true },
        new() { Key = "uni_focal_g3", Name = "UNI", Group = "UNI", Display = "UNI - Focal G3", F1 = 0.9514, Auc = 0.9958, MelFn = 3, Description = "UNI encoder with focal loss gamma=3 and melanoma weighting. Alternate UNI shortlist run with Macro F1 95.1% and Melanoma FN=3.", Available = true },
        new() { Key = "phikon_cost_sensitive_strong", Name = "Phikon", Group = "Phikon", Display = "Phikon - Cost-Sensitive Strong", F1 = 0.9424, Auc = 0.9938, MelFn = 3, Description = "Phikon encoder with stronger melanoma-miss penalty. Best fast-run Phikon shortlist model with Macro F1 94.2% and Melanoma FN=3.", Available = true },
        new() { Key = "phikon_cost_sensitive", Name = "Phikon", Group = "Phikon", Display = "Phikon - Cost-Sensitive", F1 = 0.9429, Auc = 0.9905, MelFn = 1, Description = "Phikon encoder with cost-sensitive loss that penalizes melanoma misses more heavily. Macro F1 94.3%, AUC 99.1%, Melanoma FN=1.", Available = true },
        new() { Key = "phikon_focal_g2", Name = "Phikon", Group = "Phikon", Display = "Phikon - Focal G2", F1 = 0.9205, Auc = 0.9908, MelFn = 0, Description = "Phikon encoder with focal loss gamma=2 and melanoma up-weighting. Melanoma FN=0, melanoma recall 100.0%, AUC 99.1%.", Available = true },
        new() { Key = "phikon_mel_boost_5x", Name = "Phikon", Group = "Phikon", Display = "Phikon - Mel Boost 5x", F1 = 0.9404, Auc = 0.9872, MelFn = 3, Description = "Phikon encoder with 5x melanoma class weighting. Legacy high-F1 Phikon run with Macro F1 94.0%, Melanoma FN=3.", Available = true },
        new() { Key = "phikon_mel_boost_3x", Name = "Phikon", Group = "Phikon", Display = "Phikon - Mel Boost 3x", F1 = 0.9326, Auc = 0.9869, MelFn = 1, Description = "Phikon encoder with 3x melanoma class weighting. Melanoma recall 97.4%, Macro F1 93.3%, Melanoma FN=1.", Available = true },
        new() { Key = "phikon_baseline", Name = "Phikon", Group = "Phikon", Display = "Phikon - Baseline", F1 = 0.9184, Auc = 0.9795, MelFn = 2, Description = "Phikon encoder with baseline cross-entropy training. Macro F1 91.8%, Melanoma FN=2.", Available = true },
        new() { Key = "conch_cost_sensitive_strong", Name = "CONCH", Group = "CONCH", Display = "CONCH - Cost-Sensitive Strong", F1 = 0.9323, Auc = 0.9881, MelFn = 4, Description = "CONCH encoder with strong cost-sensitive melanoma penalty. Best CONCH run with Macro F1 93.2% and Melanoma FN=4.", Available = true },
        new() { Key = "convnext_base_mel_boost_3x", Name = "ConvNeXt-Base", Group = "ConvNeXt-Base", Display = "ConvNeXt-Base - Mel Boost 3x", F1 = 0.8773, Auc = 0.9666, MelFn = 3, Description = "ConvNeXt-Base encoder with 3x melanoma class weighting. Best ConvNeXt-Base run with Macro F1 87.7% and Melanoma FN=3.", Available = true },
        new() { Key = "convnext_base_focal_g2", Name = "ConvNeXt-Base", Group = "ConvNeXt-Base", Display = "ConvNeXt-Base - Focal G2", F1 = 0.8514, Auc = 0.9668, MelFn = 1, Description = "ConvNeXt-Base encoder with focal loss gamma=2. Macro F1 85.1%, Melanoma FN=1.", Available = true },
        new() { Key = "convnext_small_mel_boost_3x", Name = "ConvNeXt-Small", Group = "ConvNeXt-Small", Display = "ConvNeXt-Small - Mel Boost 3x", F1 = 0.8632, Auc = 0.9563, MelFn = 2, Description = "ConvNeXt-Small encoder with 3x melanoma class weighting. Best ConvNeXt-Small run with Macro F1 86.3% and Melanoma FN=2.", Available = true },
        new() { Key = "convnext_small_focal_g2", Name = "ConvNeXt-Small", Group = "ConvNeXt-Small", Display = "ConvNeXt-Small - Focal G2", F1 = 0.8495, Auc = 0.9638, MelFn = 6, Description = "ConvNeXt-Small encoder with focal loss gamma=2. Macro F1 85.0%, Melanoma FN=6.", Available = true },
        new() { Key = "dinov2_base_focal_g2", Name = "DINOv2-base", Group = "DINOv2", Display = "DINOv2-Base - Focal G2", F1 = 0.8535, Auc = 0.9643, MelFn = 3, Description = "DINOv2-base encoder with focal loss gamma=2. Best DINOv2 run with Macro F1 85.3% and Melanoma FN=3.", Available = true },
        new() { Key = "dinov2_base_mel_boost_5x", Name = "DINOv2-base", Group = "DINOv2", Display = "DINOv2-Base - Mel Boost 5x", F1 = 0.8319, Auc = 0.9557, MelFn = 8, Description = "DINOv2-base encoder with 5x melanoma class weighting. Macro F1 83.2%, Melanoma FN=8.", Available = true },
        new() { Key = "resnet50_focal_g2", Name = "ResNet50", Group = "ResNet", Display = "ResNet50 - Focal G2", F1 = 0.8345, Auc = 0.9687, MelFn = 3, Description = "ResNet50 encoder with focal loss gamma=2. Macro F1 83.5%, Melanoma FN=3.", Available = true },
        new() { Key = "resnet18_focal_g2", Name = "ResNet18", Group = "ResNet", Display = "ResNet18 - Focal G2", F1 = 0.8412, Auc = 0.9588, MelFn = 6, Description = "ResNet18 encoder with focal loss gamma=2. Lightweight backbone run with Macro F1 84.1% and Melanoma FN=6.", Available = true },
    };

    public static readonly List<EnsembleInfo> Ensembles = new()
    {
        new()
        {
            Key = "gated_app_order_cheap_conf70_margin20_mel20",
            Name = "Gated Ensemble (Cost-Aware UNI -> Phikon -> CONCH)",
            Display = "Gated Cost-Aware Ensemble (UNI -> Phikon -> CONCH)",
            Description = "Sequential guarded ensemble selected from the Phase 9 feature-cost proxy profile. It starts with UNI and escalates only when confidence < 0.70, margin < 0.20, or non-melanoma prediction still has P(Melanoma) >= 0.20.",
            Models = new() { "uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong" },
            F1 = 0.9603,
            Auc = 0.9957,
            MelFn = 0,
            Gated = true,
            GatingPolicy = new()
            {
                ["name"] = "cheap_conf70_margin20_mel20",
                ["confidence_below"] = 0.70,
                ["margin_below"] = 0.20,
                ["mel_prob_at_least_if_not_mel"] = 0.20,
                ["confirm_predicted_melanoma"] = false,
                ["source"] = "results/phase9_feature_cost_profile/gating_policy_results.csv"
            }
        },
        new() { Key = "ensemble_2_best", Name = "Ensemble-2 (Best Pathology Pair)", Display = "Ensemble 2-Model (UNI + Phikon)", Description = "Average-probability ensemble of UNI - Cost-Sensitive Strong and Phikon - Cost-Sensitive Strong.", Models = new() { "uni_cost_sensitive_strong", "phikon_cost_sensitive_strong" }, F1 = 0.948, Auc = 0.995, MelFn = 0 },
        new() { Key = "ensemble_3_best", Name = "Ensemble-3 (Best Pathology Trio)", Display = "Ensemble 3-Model (UNI + Phikon + CONCH)", Description = "Average-probability ensemble of UNI - Cost-Sensitive Strong, Phikon - Cost-Sensitive Strong, and CONCH - Cost-Sensitive Strong.", Models = new() { "uni_cost_sensitive_strong", "phikon_cost_sensitive_strong", "conch_cost_sensitive_strong" }, F1 = 0.943, Auc = 0.993, MelFn = 0 },
        new() { Key = "ensemble_3", Name = "Ensemble-3 (MelFN=0)", Display = "Ensemble 3-Model (Legacy Best)", Description = "Legacy validated ensemble with historical Melanoma FN=0 validation behavior.", Models = new() { "phikon_cost_sensitive", "phikon_mel_boost_5x", "resnet50_focal_g2" }, F1 = 0.961, Auc = 0.988, MelFn = 0 },
        new() { Key = "ensemble_4", Name = "Ensemble-4 (Multi-backbone)", Display = "Ensemble 4-Model (Legacy Multi-backbone)", Description = "Legacy multi-backbone ensemble of ConvNeXt, DINOv2, and Phikon models.", Models = new() { "convnext_base_focal_g2", "dinov2_base_focal_g2", "phikon_cost_sensitive", "phikon_mel_boost_5x" }, F1 = 0.961, Auc = 0.987, MelFn = 0 },
        new() { Key = "ensemble_5", Name = "Ensemble-5 (Maximum)", Display = "Ensemble 5-Model (Legacy Maximum)", Description = "Legacy five-model ensemble with the older search-space maximum-AUC behavior.", Models = new() { "convnext_small_focal_g2", "phikon_cost_sensitive", "phikon_mel_boost_3x", "phikon_mel_boost_5x", "resnet50_focal_g2" }, F1 = 0.961, Auc = 0.989, MelFn = 0 },
    };

    public static ModelInfo? FindModel(string key) => Models.FirstOrDefault(m => m.Key == key);
    public static EnsembleInfo? FindEnsemble(string key) => Ensembles.FirstOrDefault(e => e.Key == key);
    public static bool IsKnownKey(string key) => FindModel(key) != null || FindEnsemble(key) != null;

    public static string DisplayFor(string key)
    {
        var ensemble = FindEnsemble(key);
        if (ensemble != null) return ensemble.Name;
        return FindModel(key)?.Display ?? key;
    }

    public static Dictionary<string, object?> BuildThresholdPolicy(string key)
    {
        var ensemble = FindEnsemble(key);
        if (ensemble?.Gated == true)
        {
            return new()
            {
                ["available"] = true,
                ["label"] = "SAFE-R gated: conf<0.70 | margin<0.20 | non-mel P(Mel)>=0.20",
                ["melanoma_safe_threshold"] = 0.20,
                ["selection_basis"] = "cost-aware gated ensemble",
                ["source"] = "Phase 9 gated policy"
            };
        }

        return new()
        {
            ["available"] = true,
            ["label"] = "SAFE-R melanoma review guard: non-mel P(Mel)>=0.20",
            ["melanoma_safe_threshold"] = 0.20,
            ["selection_basis"] = "post-hoc melanoma-safe selective diagnosis",
            ["source"] = ".NET compatibility safety layer"
        };
    }
}

public class ModelInfo
{
    public string Key { get; set; } = "";
    public string Name { get; set; } = "";
    public string Display { get; set; } = "";
    public string Group { get; set; } = "Other";
    public double F1 { get; set; }
    public double Auc { get; set; }
    public int MelFn { get; set; }
    public string Description { get; set; } = "";
    public bool Available { get; set; }
    public Dictionary<string, object?>? ThresholdPolicy { get; set; }
    public string? ThresholdLabel { get; set; }
}

public class AnalysisJob
{
    public string JobId { get; set; } = "";
    public string Status { get; set; } = "queued";
    public int Progress { get; set; } = 0;
    public string Message { get; set; } = "";
    public string Filename { get; set; } = "";
    public string SlidePath { get; set; } = "";
    public double FileSizeMb { get; set; }
    public string ModelKey { get; set; } = ModelCatalog.DefaultModelKey;
    public string ModelDisplay { get; set; } = "Gated Ensemble";
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    public SlideInfo? SlideInfo { get; set; }
    public AnalysisResult? Result { get; set; }
}

public class SlideInfo
{
    public long Width { get; set; }
    public long Height { get; set; }
    public double Mpp { get; set; }
    public string Vendor { get; set; } = "unknown";
    public int LevelCount { get; set; }
}

public class AnalysisResult
{
    public string Prediction { get; set; } = "";
    public int PredictionId { get; set; }
    public string RawPrediction { get; set; } = "";
    public int RawPredictionId { get; set; }
    public string PredictionKey { get; set; } = "";
    public string DecisionStatus { get; set; } = "retained";
    public Dictionary<string, double> Probabilities { get; set; } = new();
    public int NTiles { get; set; }
    public List<TopTile> TopTiles { get; set; } = new();
    public bool HeatmapAvailable { get; set; }
    public string ModelUsed { get; set; } = "";
    public string ModelKey { get; set; } = "";
    public string Timestamp { get; set; } = "";
    public List<EnsembleModelResult>? EnsembleDetails { get; set; }
    public Dictionary<string, object?>? Safety { get; set; }
    public Dictionary<string, object?>? ThresholdPolicy { get; set; }
    public Dictionary<string, object?>? Retrieval { get; set; }
    public Dictionary<string, object?>? CalculationDetails { get; set; }
    public List<HeatmapView> HeatmapViews { get; set; } = new();
    public string DefaultHeatmapView { get; set; } = "attention";
    public string? TileBaseUrl { get; set; }
    public string? ExportUrl { get; set; }
    public string? PdfReportUrl { get; set; }
}

public class EnsembleModelResult
{
    public string Model { get; set; } = "";
    public string ModelKey { get; set; } = "";
    public string Prediction { get; set; } = "";
    public Dictionary<string, double> Probabilities { get; set; } = new();
}

public class HeatmapView
{
    public string Key { get; set; } = "attention";
    public string Label { get; set; } = "Attention";
}

public class TopTile
{
    public int Rank { get; set; }
    public int TileIndex { get; set; }
    public double Attention { get; set; }
    public TileCoord Coord { get; set; } = new();
    public string ImageUrl { get; set; } = "";
}

public class TileCoord
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Level { get; set; }
    public int Size { get; set; }
    public int ReadSize { get; set; }
    public double LevelDs { get; set; }
}

public class JobStatusResponse
{
    public string JobId { get; set; } = "";
    public string Status { get; set; } = "";
    public int Progress { get; set; }
    public string Message { get; set; } = "";
    public string Filename { get; set; } = "";
    public double FileSizeMb { get; set; }
    public string ModelDisplay { get; set; } = "";
    public string CreatedAt { get; set; } = "";
    public SlideInfo? SlideInfo { get; set; }
    public AnalysisResult? Result { get; set; }
}

public class UploadResponse
{
    public string JobId { get; set; } = "";
    public string Filename { get; set; } = "";
    public double SizeMb { get; set; }
    public string Model { get; set; } = "";
}

public class HistoryItem
{
    public string JobId { get; set; } = "";
    public string Filename { get; set; } = "";
    public string Status { get; set; } = "";
    public string Model { get; set; } = "";
    public string CreatedAt { get; set; } = "";
    public AnalysisResult? Result { get; set; }
}

public class ModelsResponse
{
    public List<ModelInfo> Models { get; set; } = new();
    public List<EnsembleInfo> Ensembles { get; set; } = new();
    public EnsembleInfo? Ensemble { get; set; }
    public string Default { get; set; } = ModelCatalog.DefaultModelKey;
}

public class EnsembleInfo
{
    public string Key { get; set; } = "ensemble";
    public string Name { get; set; } = "Ensemble";
    public string Display { get; set; } = "Ensemble";
    public string Description { get; set; } = "";
    public List<string> Models { get; set; } = new();
    public double F1 { get; set; }
    public double Auc { get; set; }
    public int MelFn { get; set; }
    public bool Gated { get; set; }
    public Dictionary<string, object?>? GatingPolicy { get; set; }
    public Dictionary<string, object?>? ThresholdPolicy { get; set; }
    public string? ThresholdLabel { get; set; }
}

public class ServerInfoResponse
{
    public string Status { get; set; } = "ok";
    public int Jobs { get; set; }
    public double UploadsMb { get; set; }
    public double ResultsMb { get; set; }
    public int NModels { get; set; }
    public List<string> Classes { get; set; } = new();
    public List<string> RetrievalBanks { get; set; } = new();
}
