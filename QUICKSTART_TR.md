# (.tay) KOD MİMARİSİ — 5 Dakikalık Başlangıç

## 1. Kur

```powershell
.\install_windows.ps1
```

Kurucu çalışır Python'ı gerçekten sınar, `.tay-venv` oluşturur ve hem TAY'ı
hem TAYLANUS CPU motorunu kurar.

## 2. Durumu gör

```powershell
.\.tay-venv\Scripts\tay.exe --version
.\.tay-venv\Scripts\tay.exe doctor
```

`doctor` içinde iki ayrı bölüm vardır:

- `backends`: TAY dizilerinin NumPy/Torch/GPU durumu;
- `engines.TAYLANUS`: CFD motoru ve CPU bağımlılıkları.

## 3. Gerçek akışkan örneğini çalıştır

```powershell
.\.tay-venv\Scripts\tay.exe run examples\taylanus_vortex.tay
```

Örnek, periyodik sınır şartındaki yerelleştirilmiş sıkıştırılamaz girdabı
32³-eşdeğer çözünürlükte `t=0.20` zamanına kadar çözer.

Temel TAY kodu:

```tay
BACKEND NUMPY
ENGINE TAYLANUS
RESOLUTION 32
DT 0.005
TEND 0.20
MODE AUTO
OUTPUT "../outputs/taylanus_vortex"
RUN TAYLANUS
```

## 4. Sonuçları aç

`outputs\taylanus_vortex` altında:

- `diagnostics.json`: ağ, temsil, DOF, süre, enerji, diverjans, önbellek ve
  referans hatası;
- `kinetic_energy.csv`: her adımın gerçek enerji değeri;
- `velocity_final.npy`: son hız alanı;
- `mesh_levels.npy`: adaptif ağ seviyeleri;
- `conservative_divergence.npy`: hücrelere eşlenen konservatif diverjans;
- altı adet açıklamalı PNG grafik.

## Mod seçimi

| MODE | Davranış |
|---|---|
| `AUTO` | İki temsili ölçer, soğuk çalışma maliyetine göre seçer |
| `FAST` | `SUBFACE_SPARSE` |
| `COMPACT` | `MODAL_STREAM` |
| `SUBFACE_SPARSE` | Temsili doğrudan seçer |
| `MODAL_STREAM` | Temsili doğrudan seçer |

`COMPACT` durum tasarrufu sağlar; CPU'da daha hızlı olduğu iddia edilmez.

## Normal TAY başlangıcı

```powershell
.\.tay-venv\Scripts\tay.exe init benim-projem
.\.tay-venv\Scripts\tay.exe run benim-projem\hello.tay
.\.tay-venv\Scripts\tay.exe notebook benim-projem\explore.taynb
.\.tay-venv\Scripts\tay.exe repl
```

## GPU konusunda net sınır

`BACKEND GPU`, normal TAY dizi kodu için CUDA/PyTorch yoludur. TAYLANUS v3
GPU kodu doğrulanmadığı için CFD motoru GPU veya Torch isteğinde hata verir;
CPU'ya sessizce geçmez.

Ayrıntılar: `docs\TAYLANUS_ENGINE.md`.
