# PleGL Sculpt

Software digital sculpting 3D untuk desktop yang fokus pada satu hal: sculpting yang cepat, stabil, dan mudah dipelajari.

Status: **Phase 6e (Sculpt layers)**. Tujuh brush (Draw, Clay, Smooth, Grab, Inflate, Flatten, Crease), empat jenis falloff, tekanan pen, undo/redo, simetri X, dan quad remesh (voxel remesh lalu optimasi valence 4 sampai sekitar 99%, relaksasi, dan proyeksi ke permukaan asli) sudah jalan di atas engine Phase 0 (mesh, BVH, renderer, import/export OBJ). Project bisa disimpan dan dibuka sebagai `.psculpt`, dengan autosave tiap 5 menit dan pemulihan setelah crash. Mask melindungi area dari semua brush, bisa dicat, dibalik, dihapus, diperhalus, dan dipertajam, lalu ikut tersimpan di project dan ikut pindah saat remesh. Dynamic topology (Ctrl+D) menambah segitiga di tempat yang sedang di-sculpt dan menggabungkannya di tempat yang tidak butuh detail, dengan ukuran detail dalam pixel layar atau satuan objek, dan tetap bisa di-undo per stroke. Face set mengelompokkan poligon dengan warna untuk membatasi brush, membuat mask, atau menyembunyikan bagian model. Multiresolution (Ctrl+Page Up) menambah level subdivisi Catmull-Clark: bentuk besar dipahat di level rendah, detail di level tinggi, dan perubahan di satu level ikut menyebar ke level lain. Sculpt layer (Ctrl+L) menyimpan detail secara non-destruktif: setiap layer bisa dikecilkan, dibalik, disembunyikan, digabung, atau dihapus kapan saja, dan ikut tersimpan di project.

## Stack

| Lapisan | Pilihan |
| --- | --- |
| Bahasa | C++20, CMake 3.24+ |
| Window dan input (termasuk pressure pen) | SDL3 |
| Rendering | OpenGL 4.5+ (direct state access), loader glad |
| UI | Dear ImGui (docking) + ImGuizmo |
| Matematika | glm |
| Test | doctest |

Semua dependensi diunduh otomatis oleh CMake (`cmake/Dependencies.cmake`) dan dikunci ke versi tertentu.

## Build

### Windows (Visual Studio 2022 atau lebih baru)

```bat
cmake --preset msvc
cmake --build --preset msvc
ctest --preset msvc
build\msvc\Release\PleGLSculpt.exe
```

Atau buka folder repo langsung di Visual Studio; preset akan terdeteksi otomatis.

### Linux

```bash
sudo apt-get install ninja-build libx11-dev libxext-dev libxcursor-dev libxi-dev libxrandr-dev \
  libxss-dev libxtst-dev libxfixes-dev libxinerama-dev libxkbcommon-dev libwayland-dev \
  wayland-protocols libdecor-0-dev libgl-dev libegl-dev
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/PleGLSculpt
```

### Hanya engine (tanpa window, untuk test dan benchmark)

```bash
cmake --preset engine-only
cmake --build --preset engine-only
./build/engine/plegl_bench
```

## Kontrol

| Aksi | Input |
| --- | --- |
| Orbit (berputar di sekitar permukaan di bawah cursor) | Alt + drag kiri |
| Pan | Alt + drag tengah |
| Zoom | Alt + drag kanan, atau scroll |
| Pilih objek | Klik kiri |
| Move / Rotate / Scale (Object mode) | G / R / S |
| Duplicate (Object mode) | Shift + D |
| Delete (Object mode) | X atau Delete |
| Frame objek terpilih | Home |
| Object / Sculpt mode | Tab |
| Sculpt (di Sculpt mode) | Drag kiri |
| Smooth sementara | Shift + drag kiri |
| Balik arah brush (Add/Subtract) | Ctrl + drag kiri |
| Pilih brush Draw / Clay / Smooth / Grab | D / C / S / G |
| Pilih brush Inflate / Flatten / Crease | I / T / Shift + C |
| Brush Mask | M (drag untuk mengecat, Ctrl + drag atau ujung penghapus pen untuk menghapus, Shift + drag untuk menghaluskan mask) |
| Invert mask | Ctrl + I (di Sculpt mode) |
| Hapus semua mask / mask semua | Alt + M / Alt + Shift + M (di kedua mode) |
| Blur / Sharpen mask | Alt + B / Alt + Shift + B (di kedua mode) |
| Brush Face Set | P (setiap stroke mengecat set baru, Ctrl + drag memperluas set di bawah cursor) |
| Sembunyikan face set di bawah cursor | H (di Sculpt mode) |
| Tampilkan hanya face set di bawah cursor | Shift + H (tekan lagi untuk menampilkan semua) |
| Tampilkan semua yang tersembunyi | Alt + H (di kedua mode) |
| Mask face set di bawah cursor | Shift + M (di Sculpt mode) |
| Radius brush | [ dan ], atau tahan F lalu geser mouse ke samping |
| Dynamic topology on/off | Ctrl + D (di Sculpt mode) |
| Ukuran detail dynamic topology | Tahan R lalu geser mouse ke samping (di Sculpt mode) |
| Simetri X on/off | X (di Sculpt mode) |
| Remesh objek terpilih | Ctrl + R (menghapus level subdivisi dan memanggang layer; undo mengembalikannya) |
| Layer sculpt baru (stroke masuk ke layer aktif) | Ctrl + L |
| Tampilkan / sembunyikan layer aktif | L (di Sculpt mode) |
| Brush Erase Layer (menghapus detail layer aktif) | E (di Sculpt mode) |
| Tampilkan hanya satu layer | Alt + klik ikon mata di panel layer (lagi: tampilkan semua) |
| Subdivide (tambah level multiresolution di atas) | Ctrl + Page Up |
| Level subdivisi naik / turun | Page Up / Page Down |
| Level tertinggi / terendah | Shift + Page Up / Shift + Page Down |
| Undo / Redo | Ctrl + Z / Ctrl + Shift + Z atau Ctrl + Y |
| Project baru / buka / simpan / simpan sebagai | Ctrl + N / Ctrl + O / Ctrl + S / Ctrl + Shift + S |
| Import / Export OBJ | Ctrl + Shift + I (Ctrl + I di Object mode) / Ctrl + E |
| Uji update GPU parsial | B di Object mode (menonjolkan permukaan di bawah cursor) |

File project (`.psculpt`) dan OBJ juga bisa dibuka dengan menaruhnya sebagai argumen (`PleGLSculpt kepala.psculpt`) atau dengan drag-and-drop ke jendela.

## Struktur

```
src/engine/   Library engine tanpa UI (diuji headless)
  core/       Tipe dasar, parallelFor, timer
  mesh/       Half-edge mesh berbasis index, primitive, validator, operasi topologi, data sculpt layer
  spatial/    BVH (node daun = unit kerja engine), raycast, query bola, titik terdekat
  io/         Import dan export OBJ, project file .psculpt
  scene/      Objek, transform, picking
  sculpt/     Brush, stroke sampler, sculptor (dab, simetri), dynamic topology, operasi mask, face set, dan layer, undo per daun BVH
  render/     Indeks gambar per daun BVH (tanpa OpenGL, diuji headless)
  remesh/     Grid voxel (SDF), Surface Nets quad, voxel remesh, quad remesh
  multires/   Level subdivisi Catmull-Clark, sinkronisasi antar level, undo, penyimpanan
src/app/      Aplikasi: SDL3, renderer OpenGL, kamera, UI
tests/        Unit test (doctest)
bench/        Benchmark engine pada mesh 50K sampai 1M vertex
docs/         Catatan arsitektur
```

Lihat [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) untuk keputusan desain utama.
