using SkinSight.Models;

namespace SkinSight.Services;

/// <summary>
/// Thread-safe LRU cache for OpenSlide + DeepZoomGenerator pairs.
/// Equivalent to Python SlideCache class.
/// </summary>
public class SlideCache : IDisposable
{
    private readonly int _maxSize;
    private readonly SkinSightConfig _config;
    private readonly LinkedList<string> _order = new();           // LRU order
    private readonly Dictionary<string, CacheEntry> _cache = new();
    private readonly object _lock = new();

    private record CacheEntry(
        OpenSlideInterop.Slide Slide,
        DeepZoomGenerator DeepZoom,
        LinkedListNode<string> Node);

    public SlideCache(SkinSightConfig config)
    {
        _maxSize = config.SlideCacheSize;
        _config = config;
    }

    /// <summary>Get or open a slide. Returns (Slide, DeepZoomGenerator).</summary>
    public (OpenSlideInterop.Slide slide, DeepZoomGenerator dz) Get(string path)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(path, out var entry))
            {
                // Move to end (most recently used)
                _order.Remove(entry.Node);
                _order.AddLast(entry.Node);
                return (entry.Slide, entry.DeepZoom);
            }

            // Open new slide
            var slide = new OpenSlideInterop.Slide(path);
            var dz = new DeepZoomGenerator(slide,
                tileSize: _config.DziTileSize,
                overlap: _config.DziOverlap);

            var node = _order.AddLast(path);
            _cache[path] = new CacheEntry(slide, dz, node);

            // Evict oldest if over capacity
            while (_cache.Count > _maxSize)
            {
                var oldestNode = _order.First!;
                var oldPath = oldestNode.Value;
                if (_cache.TryGetValue(oldPath, out var old))
                {
                    old.DeepZoom.Dispose();
                    old.Slide.Dispose();
                    _cache.Remove(oldPath);
                }
                _order.RemoveFirst();
            }

            return (slide, dz);
        }
    }

    /// <summary>Close and remove a specific slide from cache.</summary>
    public void Remove(string path)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(path, out var entry))
            {
                _order.Remove(entry.Node);
                entry.DeepZoom.Dispose();
                entry.Slide.Dispose();
                _cache.Remove(path);
            }
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            foreach (var entry in _cache.Values)
            {
                entry.DeepZoom.Dispose();
                entry.Slide.Dispose();
            }
            _cache.Clear();
            _order.Clear();
        }
    }

    public void Dispose()
    {
        Clear();
        GC.SuppressFinalize(this);
    }
}
