using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SkinSight.Models;

namespace SkinSight.Services;

/// <summary>
/// Deep Zoom Image (DZI) generator – serves tile pyramid on-demand.
/// Equivalent to Python openslide.deepzoom.DeepZoomGenerator.
/// </summary>
public class DeepZoomGenerator : IDisposable
{
    private readonly OpenSlideInterop.Slide _slide;
    private readonly int _tileSize;
    private readonly int _overlap;

    public int LevelCount { get; }
    public (long w, long h)[] LevelDimensions { get; }
    public (int cols, int rows)[] LevelTiles { get; }

    public DeepZoomGenerator(OpenSlideInterop.Slide slide, int tileSize = 254, int overlap = 1)
    {
        _slide = slide;
        _tileSize = tileSize;
        _overlap = overlap;

        // Build DZI pyramid levels
        // DZI levels go from 1x1 up to full resolution
        long w = slide.Width;
        long h = slide.Height;

        // Calculate total DZI levels
        int maxDim = (int)Math.Max(w, h);
        int dziLevelCount = (int)Math.Ceiling(Math.Log2(maxDim)) + 1;
        LevelCount = dziLevelCount;

        LevelDimensions = new (long w, long h)[dziLevelCount];
        LevelTiles = new (int cols, int rows)[dziLevelCount];

        for (int i = 0; i < dziLevelCount; i++)
        {
            // DZI level 0 = 1x1, level max = full resolution
            double scale = Math.Pow(2, dziLevelCount - 1 - i);
            long lw = Math.Max(1, (long)Math.Ceiling(w / scale));
            long lh = Math.Max(1, (long)Math.Ceiling(h / scale));
            LevelDimensions[i] = (lw, lh);

            int cols = (int)Math.Ceiling((double)lw / _tileSize);
            int rows = (int)Math.Ceiling((double)lh / _tileSize);
            LevelTiles[i] = (cols, rows);
        }
    }

    /// <summary>Get a single DZI tile as an ImageSharp image</summary>
    public SixLabors.ImageSharp.Image GetTile(int level, int col, int row)
    {
        if (level < 0 || level >= LevelCount)
            throw new ArgumentOutOfRangeException(nameof(level));

        var (levelW, levelH) = LevelDimensions[level];
        var (maxCols, maxRows) = LevelTiles[level];

        if (col < 0 || col >= maxCols || row < 0 || row >= maxRows)
            throw new ArgumentOutOfRangeException("Tile coordinates out of range");

        // Calculate tile boundaries in DZI level coordinates
        int x = col * _tileSize;
        int y = row * _tileSize;

        // Add overlap
        int x0 = col == 0 ? 0 : x - _overlap;
        int y0 = row == 0 ? 0 : y - _overlap;
        int x1 = Math.Min((int)levelW, x + _tileSize + _overlap);
        int y1 = Math.Min((int)levelH, y + _tileSize + _overlap);
        int tileW = x1 - x0;
        int tileH = y1 - y0;

        // Map DZI level coordinates to OpenSlide level-0 coordinates
        double scale = Math.Pow(2, LevelCount - 1 - level);
        long slideX = (long)(x0 * scale);
        long slideY = (long)(y0 * scale);

        // Find best OpenSlide level for this zoom
        int osLevel = _slide.GetBestLevelForDownsample(scale);
        double osDs = _slide.GetLevelDownsample(osLevel);

        // Compute read size in the chosen OpenSlide level
        int readW = (int)Math.Ceiling(tileW * scale / osDs);
        int readH = (int)Math.Ceiling(tileH * scale / osDs);

        // Clamp to slide dimensions
        readW = Math.Max(1, readW);
        readH = Math.Max(1, readH);

        var region = _slide.ReadRegion(slideX, slideY, osLevel, readW, readH);

        // Resize to target DZI tile size if OpenSlide level doesn't match exactly
        if (region.Width != tileW || region.Height != tileH)
        {
            region.Mutate(ctx => ctx.Resize(tileW, tileH));
        }

        return region;
    }

    public void Dispose()
    {
        // Note: we don't own the slide, so we don't dispose it
        GC.SuppressFinalize(this);
    }
}
