# PleGL Sculpt

Software digital sculpting 3D untuk desktop yang fokus pada satu hal: sculpting yang cepat, stabil, dan mudah dipelajari.

Status: **Phase 5 (Project file)**. Tujuh brush (Draw, Clay, Smooth, Grab, Inflate, Flatten, Crease), empat jenis falloff, tekanan pen, undo/redo, simetri X, dan quad remesh (voxel remesh lalu optimasi valence 4 sampai sekitar 99%, relaksasi, dan proyeksi ke permukaan asli) sudah jalan di atas engine Phase 0 (mesh, BVH, renderer, import/export OBJ). Project bisa disimpan dan dibuka sebagai `.psculpt`, dengan autosave tiap 5 menit dan pemulihan setelah crash.

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
| Radius brush | [ dan ], atau tahan F lalu geser mouse ke samping |
| Simetri X on/off | X (di Sculpt mode) |
| Remesh objek terpilih | Ctrl + R |
| Undo / Redo | Ctrl + Z / Ctrl + Shift + Z atau Ctrl + Y |
| Project baru / buka / simpan / simpan sebagai | Ctrl + N / Ctrl + O / Ctrl + S / Ctrl + Shift + S |
| Import / Export OBJ | Ctrl + I / Ctrl + E |
| Uji update GPU parsial | B (menonjolkan permukaan di bawah cursor) |

File project (`.psculpt`) dan OBJ juga bisa dibuka dengan menaruhnya sebagai argumen (`PleGLSculpt kepala.psculpt`) atau dengan drag-and-drop ke jendela.

## Struktur

```
src/engine/   Library engine tanpa UI (diuji headless)
  core/       Tipe dasar, parallelFor, timer
  mesh/       Half-edge mesh berbasis index, primitive, validator, operasi topologi
  spatial/    BVH (node daun = unit kerja engine), raycast, query bola, titik terdekat
  io/         Import dan export OBJ, project file .psculpt
  scene/      Objek, transform, picking
  sculpt/     Brush, stroke sampler, sculptor (dab, simetri), undo per daun BVH
  remesh/     Grid voxel (SDF), Surface Nets quad, voxel remesh, quad remesh
src/app/      Aplikasi: SDL3, renderer OpenGL, kamera, UI
tests/        Unit test (doctest)
bench/        Benchmark engine pada mesh 50K sampai 1M vertex
docs/         Catatan arsitektur
```

Lihat [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) untuk keputusan desain utama.
