# SkinSight .NET – ASP.NET Core Port

Bu klasör, Python Flask uygulamasının (`../app/`) birebir .NET 8 (ASP.NET Core) portudur.

## Mimari Karşılaştırma

| Bileşen | Python (Flask) | .NET (ASP.NET Core) |
|---------|---------------|-------------------|
| Web Framework | Flask + CORS | ASP.NET Core Minimal API |
| Slide Reading | openslide-python | P/Invoke → libopenslide.so |
| DZI Tiles | openslide.deepzoom | Custom DeepZoomGenerator |
| Image Processing | Pillow + OpenCV | SixLabors.ImageSharp |
| ML Inference | PyTorch (runtime) | ONNX Runtime |
| Heatmap | NumPy + OpenCV | Pure C# (float arrays) |
| Concurrency | threading | Task.Run + ConcurrentDictionary |

## Ön Gereksinimler

### 1. .NET 8 SDK
```bash
# Ubuntu
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0

# Veya apt ile:
sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0
```

### 2. OpenSlide Native Library
```bash
# Ubuntu/Debian – BU ŞART!
sudo apt-get install -y libopenslide0 libopenslide-dev

# macOS
brew install openslide
```

### 3. (Opsiyonel) ONNX Model Dosyaları
PyTorch modellerini ONNX'e dönüştürmek için `export_to_onnx.py` scriptini çalıştırın:
```bash
cd ..
python app_dotnet/export_to_onnx.py
```

## Çalıştırma

```bash
cd app_dotnet
dotnet restore
dotnet run
```

Sunucu `http://localhost:5001` adresinde başlar.

## OpenSlide Uyumluluğu

**Çalışır mı?** Evet! OpenSlide bir C kütüphanesidir ve .NET'ten P/Invoke ile doğrudan çağrılabilir:

- `libopenslide.so` (Linux) veya `openslide.dll` (Windows) sistem kütüphaneleri kullanılır
- Python `openslide-python` paketi de aslında aynı C kütüphanesini çağırır (ctypes ile)
- .NET versiyonu `DllImport` kullanarak aynı native fonksiyonları çağırır
- Desteklenen formatlar aynıdır: `.svs`, `.tif`, `.ndpi`, `.mrxs`, `.scn`

### DZI (Deep Zoom Image) Desteği

Python versiyonunda `openslide.deepzoom.DeepZoomGenerator` kullanılır.
.NET versiyonunda bunu sıfırdan yazdık (`Services/DeepZoomGenerator.cs`):
- Aynı tile pyramid mantığı
- Aynı overlap ve tile size parametreleri
- On-demand tile serving (disk'e yazmadan)

## ML Inference Farkı

Python versiyonu PyTorch modellerini doğrudan yükler.  
.NET versiyonu **ONNX Runtime** kullanır:

1. Önce modelleri ONNX'e export etmeniz gerekir (bir kerelik)
2. ONNX Runtime platform-bağımsız ve hızlıdır
3. Model yoksa dummy prediction döner (Python versiyonuyla aynı davranış)

## Dosya Yapısı

```
app_dotnet/
├── SkinSight.csproj          # Proje dosyası
├── Program.cs                # Ana giriş + tüm API endpoint'leri
├── appsettings.json          # Konfigürasyon
├── Models/
│   └── AnalysisModels.cs     # Data modelleri
├── Services/
│   ├── OpenSlideInterop.cs   # OpenSlide P/Invoke bindings
│   ├── DeepZoomGenerator.cs  # DZI tile pyramid generator
│   ├── SlideCache.cs         # LRU slide handle cache
│   └── AnalysisService.cs    # Analiz pipeline
├── wwwroot/                  # Static dosyalar (= app/static/)
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── export_to_onnx.py         # PyTorch → ONNX dönüşüm scripti
└── README.md
```

## API Endpoint'leri (Python ile Birebir Aynı)

| Method | Path | Açıklama |
|--------|------|----------|
| POST | `/api/upload` | WSI dosyası yükle ve analiz başlat |
| GET | `/api/status/{jobId}` | Analiz durumu sorgula |
| GET | `/api/results/{jobId}/dzi/slide.dzi` | DZI descriptor |
| GET | `/api/results/{jobId}/dzi/slide_files/{level}/{tile}` | DZI tile |
| GET | `/api/results/{jobId}/heatmap` | Heatmap overlay |
| GET | `/api/results/{jobId}/heatmap_only` | Transparent heatmap |
| GET | `/api/results/{jobId}/thumbnail` | Slide thumbnail |
| GET | `/api/results/{jobId}/tiles/{filename}` | Analysis tile |
| GET | `/api/results/{jobId}/export` | JSON export |
| POST | `/api/results/{jobId}/delete` | Sonuçları sil |
| GET | `/api/history` | Tüm analizleri listele |
| GET | `/api/demo` | Demo sonuçları |
| GET | `/api/info` | Sunucu bilgisi |

## Performans Notları

- .NET'in AOT compilation ve JIT optimizasyonları sayesinde tile serving Python'dan daha hızlı olabilir
- ONNX Runtime, PyTorch inference'dan genelde 2-3x daha hızlıdır
- `ConcurrentDictionary` kullanımı, Python'daki `threading.Lock` yerine daha iyi concurrency sağlar
- Kestrel web sunucusu, Flask/gunicorn'dan daha yüksek throughput'a sahiptir
