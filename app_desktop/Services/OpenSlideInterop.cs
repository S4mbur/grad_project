using System.Runtime.InteropServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SkinSight.Services;

/// <summary>
/// P/Invoke bindings for OpenSlide (C library for whole-slide images).
/// 
/// PREREQUISITES (Windows):
///   1. Download OpenSlide Windows binaries from https://openslide.org/download/
///   2. Extract and copy all .dll files to the app_desktop/openslide/ folder
///      (they will be auto-copied to the output directory on build)
///   3. Or place them next to SkinSight.exe in the output folder
///
///   Required DLLs: libopenslide-1.dll (OpenSlide 4.x, self-contained)
/// </summary>
public static class OpenSlideInterop
{
    // Windows: openslide 4.x uses "libopenslide-1.dll"
    private const string LibName = "libopenslide-1";

    static OpenSlideInterop()
    {
        NativeLibrary.SetDllImportResolver(typeof(OpenSlideInterop).Assembly, (libraryName, assembly, searchPath) =>
        {
            if (!libraryName.Equals(LibName, StringComparison.OrdinalIgnoreCase))
                return IntPtr.Zero;

            var candidates = new[]
            {
                Path.Combine(AppContext.BaseDirectory, "libopenslide-1.dll"),
                Path.Combine(AppContext.BaseDirectory, "openslide", "libopenslide-1.dll")
            };

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                    return NativeLibrary.Load(candidate);
            }

            return IntPtr.Zero;
        });
    }

    // ─── Native function imports ────────────────────────────────────

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr openslide_detect_vendor(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr openslide_open(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string filename);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void openslide_close(IntPtr osr);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void openslide_get_level0_dimensions(
        IntPtr osr, out long w, out long h);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int openslide_get_level_count(IntPtr osr);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void openslide_get_level_dimensions(
        IntPtr osr, int level, out long w, out long h);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern double openslide_get_level_downsample(
        IntPtr osr, int level);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int openslide_get_best_level_for_downsample(
        IntPtr osr, double downsample);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void openslide_read_region(
        IntPtr osr, IntPtr dest, long x, long y, int level, long w, long h);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr openslide_get_error(IntPtr osr);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr openslide_get_property_value(
        IntPtr osr, [MarshalAs(UnmanagedType.LPUTF8Str)] string name);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr openslide_get_property_names(IntPtr osr);

    // ─── Safe wrapper class ────────────────────────────────────────

    /// <summary>
    /// Managed wrapper around an OpenSlide handle.
    /// Implements IDisposable to ensure native resources are freed.
    /// </summary>
    public class Slide : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed = false;

        public long Width { get; }
        public long Height { get; }
        public int LevelCount { get; }

        public Slide(string filename)
        {
            if (!File.Exists(filename))
                throw new FileNotFoundException($"Slide not found: {filename}");

            _handle = openslide_open(filename);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open slide: {filename}");

            var error = GetError();
            if (error != null)
                throw new InvalidOperationException($"OpenSlide error: {error}");

            openslide_get_level0_dimensions(_handle, out long w, out long h);
            Width = w;
            Height = h;
            LevelCount = openslide_get_level_count(_handle);
        }

        public (long w, long h) GetLevelDimensions(int level)
        {
            CheckDisposed();
            openslide_get_level_dimensions(_handle, level, out long w, out long h);
            return (w, h);
        }

        public double GetLevelDownsample(int level)
        {
            CheckDisposed();
            return openslide_get_level_downsample(_handle, level);
        }

        public int GetBestLevelForDownsample(double downsample)
        {
            CheckDisposed();
            return openslide_get_best_level_for_downsample(_handle, downsample);
        }

        /// <summary>Read a region from the slide and return as ImageSharp Image.</summary>
        public Image<Rgba32> ReadRegion(long x, long y, int level, int w, int h)
        {
            CheckDisposed();
            int pixelCount = w * h;
            int bufferSize = pixelCount * 4; // ARGB, 4 bytes per pixel

            IntPtr buffer = Marshal.AllocHGlobal(bufferSize);
            try
            {
                openslide_read_region(_handle, buffer, x, y, level, w, h);

                var error = GetError();
                if (error != null)
                    throw new InvalidOperationException($"ReadRegion error: {error}");

                // OpenSlide returns pre-multiplied ARGB pixels
                var image = new Image<Rgba32>(w, h);
                var pixelData = new byte[bufferSize];
                Marshal.Copy(buffer, pixelData, 0, bufferSize);

                image.ProcessPixelRows(accessor =>
                {
                    for (int row = 0; row < h; row++)
                    {
                        var pixelRow = accessor.GetRowSpan(row);
                        for (int col = 0; col < w; col++)
                        {
                            int offset = (row * w + col) * 4;
                            byte b = pixelData[offset + 0];
                            byte g = pixelData[offset + 1];
                            byte r = pixelData[offset + 2];
                            byte a = pixelData[offset + 3];

                            // Un-premultiply alpha
                            if (a > 0 && a < 255)
                            {
                                r = (byte)Math.Min(255, r * 255 / a);
                                g = (byte)Math.Min(255, g * 255 / a);
                                b = (byte)Math.Min(255, b * 255 / a);
                            }
                            pixelRow[col] = new Rgba32(r, g, b, a);
                        }
                    }
                });

                return image;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>Get a slide property (e.g. "openslide.mpp-x")</summary>
        public string? GetProperty(string name)
        {
            CheckDisposed();
            IntPtr ptr = openslide_get_property_value(_handle, name);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        /// <summary>Get a thumbnail of the slide</summary>
        public Image<Rgba32> GetThumbnail(int maxWidth, int maxHeight)
        {
            // Use the lowest resolution level
            int thumbLevel = LevelCount - 1;
            var (lw, lh) = GetLevelDimensions(thumbLevel);

            // If still too big, find a better level
            for (int lv = 0; lv < LevelCount; lv++)
            {
                var (w, h) = GetLevelDimensions(lv);
                if (w <= maxWidth && h <= maxHeight)
                {
                    thumbLevel = lv;
                    (lw, lh) = (w, h);
                    break;
                }
            }

            var img = ReadRegion(0, 0, thumbLevel, (int)lw, (int)lh);

            // Resize if still too large
            if (lw > maxWidth || lh > maxHeight)
            {
                double scale = Math.Min((double)maxWidth / lw, (double)maxHeight / lh);
                int newW = (int)(lw * scale);
                int newH = (int)(lh * scale);
                img.Mutate(x => x.Resize(newW, newH));
            }

            return img;
        }

        private string? GetError()
        {
            IntPtr errPtr = openslide_get_error(_handle);
            return errPtr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(errPtr);
        }

        private void CheckDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Slide));
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                openslide_close(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Slide() => Dispose();
    }

    /// <summary>Detect the slide vendor (format)</summary>
    public static string? DetectVendor(string filename)
    {
        IntPtr ptr = openslide_detect_vendor(filename);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
    }
}
