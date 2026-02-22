namespace SkinSight.Models;

/// <summary>Configuration matching Python AppConfig</summary>
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
    public string OnnxModelPath { get; set; } = "../data/mil_results/best_model.onnx";
    public string OnnxEncoderPath { get; set; } = "../data/mil_results/encoder.onnx";
}

/// <summary>4-class mapping</summary>
public static class ClassInfo
{
    public static readonly Dictionary<int, string> Names = new()
    {
        { 0, "Normal/Benign" }, { 1, "BCC" }, { 2, "SCC" }, { 3, "Melanoma" }
    };
    public static readonly int Count = 4;
}

/// <summary>Registered MIL model info</summary>
public class ModelInfo
{
    public string Key { get; set; } = "";
    public string Name { get; set; } = "";
    public string Display { get; set; } = "";
    public double F1 { get; set; }
    public double Auc { get; set; }
    public string Description { get; set; } = "";
    public bool Available { get; set; }
}

/// <summary>Tracks an analysis job</summary>
public class AnalysisJob
{
    public string JobId { get; set; } = "";
    public string Status { get; set; } = "queued";       // queued, processing, completed, error
    public int Progress { get; set; } = 0;
    public string Message { get; set; } = "";
    public string Filename { get; set; } = "";
    public string SlidePath { get; set; } = "";
    public double FileSizeMb { get; set; }
    public string ModelKey { get; set; } = "phikon";
    public string ModelDisplay { get; set; } = "Phikon";
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
    public Dictionary<string, double> Probabilities { get; set; } = new();
    public int NTiles { get; set; }
    public List<TopTile> TopTiles { get; set; } = new();
    public bool HeatmapAvailable { get; set; }
    public string ModelUsed { get; set; } = "";
    public string ModelKey { get; set; } = "";
    public string Timestamp { get; set; } = "";
    public List<EnsembleModelResult>? EnsembleDetails { get; set; }
}

public class EnsembleModelResult
{
    public string Model { get; set; } = "";
    public string Prediction { get; set; } = "";
    public Dictionary<string, double> Probabilities { get; set; } = new();
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

// DTOs for API responses (exclude internal fields)
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
    public EnsembleInfo? Ensemble { get; set; }
    public string Default { get; set; } = "phikon";
}

public class EnsembleInfo
{
    public string Key { get; set; } = "ensemble";
    public string Name { get; set; } = "Ensemble";
    public string Display { get; set; } = "Ensemble (Top-3 Models)";
    public string Description { get; set; } = "";
    public List<string> Models { get; set; } = new();
}

public class ServerInfoResponse
{
    public string Status { get; set; } = "ok";
    public int Jobs { get; set; }
    public double UploadsMb { get; set; }
    public double ResultsMb { get; set; }
    public int NModels { get; set; }
    public List<string> Classes { get; set; } = new();
}
