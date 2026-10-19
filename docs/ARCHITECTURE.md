# Arsitektur

Dokumen ini merangkum keputusan yang sudah diterapkan di kode. Rencana lengkap Phase 0 dan Phase 1 ada di dokumen rencana teknis proyek.

## Engine terpisah dari aplikasi

`plegl_engine` adalah library statis tanpa window, GPU, atau UI. Semua yang ada di dalamnya bisa diuji dan di-benchmark secara headless (`plegl_tests`, `plegl_bench`). Aplikasi (`src/app`) hanya lapisan tipis: input, kamera, renderer, dan panel.

## Mesh: half-edge berbasis index (`mesh/Mesh.h`)

- Topologi adalah array `int32`: `heNext`, `heTwin`, `heVert`, `heFace`, ditambah `vertHe` dan `faceHe`. Tidak ada pointer, jadi mesh murah disalin, diserialisasi, dan dikirim ke thread lain.
- Atribut disimpan sebagai struct-of-arrays (`positions`, `normals`). Jalur panas brush hanya menyentuh array ini.
- Edge terbuka ditandai `heTwin == -1`, tanpa loop boundary eksplisit.
- Face boleh segitiga, quad, atau n-gon. Output voxel remesh nanti quad-dominant.
- `buildMesh` membangun twin lewat adjacency per vertex (linear terhadap jumlah half-edge) dan melaporkan edge non-manifold, vertex non-manifold, dan face degenerate.
- `validate()` memeriksa semua invarian topologi. Dipakai di test dan nanti di build debug setiap selesai operasi topologi.

## BVH: node daun sebagai unit kerja (`spatial/Bvh.h`)

- Face dikelompokkan ke node daun berisi paling banyak 1.024 face (median split pada sumbu terpanjang).
- `Bvh::build` **mengurutkan ulang mesh** sehingga setiap daun memiliki rentang face dan rentang vertex yang bersambung. Vertex dimiliki oleh daun pertama yang memakainya.
- Karena itu satu daun bisa menjadi unit untuk raycast, query brush, upload GPU parsial (`glNamedBufferSubData` per rentang vertex), snapshot undo, dan pembagian kerja antar thread.
- Setelah vertex bergerak cukup `refitLeaves` (murah). Rebuild penuh hanya setelah topologi berubah.
- Raycast menelusuri node dari depan ke belakang dan berhenti begitu node berikutnya lebih jauh dari hit terbaik.

## Renderer (`src/app/Renderer.*`)

- Satu buffer posisi dan satu buffer normal per objek, urutannya sama dengan mesh. Daun yang ditandai dirty diunggah per rentang, bukan seluruh buffer.
- Wireframe memakai edge poligon asli (bukan diagonal segitiga), jadi quad tampil sebagai quad.
- Matcap dibuat secara prosedural saat start, tanpa file aset.

## Sculpting (`sculpt/`)

Alur satu stroke: event mouse/pen → `StrokeSampler` (jarak antar dab dalam pixel layar, tekanan diinterpolasi) → raycast ke objek → `Sculptor::dab`.

Setiap dab:

1. Query bola BVH memberi daun yang tersentuh.
2. Daun itu disalin ke snapshot undo sebelum brush menyentuhnya (sekali per stroke per daun).
3. Brush memindahkan posisi secara paralel per daun. Kontraknya: vertex hanya bergerak di dalam bola dab.
4. Normal dihitung ulang hanya untuk vertex dari face yang berada dalam 1,25 × radius, lalu bounds daun di-refit dan daun ditandai dirty untuk upload GPU parsial.
5. Dengan simetri X, dab cermin dijalankan di (−x, y, z), kecuali dab sudah menyentuh bidang tengah.

Radius brush ditentukan dalam pixel layar, lalu diubah ke satuan dunia di titik kena, jadi ukuran brush terasa sama di layar pada zoom berapa pun. Tekanan pen bisa memengaruhi strength, radius, keduanya, atau tidak sama sekali.

Brush yang ada:

| Brush | Cara kerja |
| --- | --- |
| Draw | Mendorong vertex sepanjang normal area. |
| Clay | Menarik vertex di bawah bidang (sedikit di atas permukaan) ke bidang itu, jadi volume bertambah sekaligus rata. |
| Smooth | Laplacian ke rata-rata tetangga; vertex border hanya dirata-rata dengan tetangga border. |
| Inflate | Mendorong tiap vertex sepanjang normalnya sendiri. |
| Flatten | Menarik vertex ke bidang yang melewati pusat area. |
| Crease | Mendorong ke dalam seperti Draw sambil menjepit vertex ke pusat brush. |
| Grab | Menangkap vertex di dalam radius saat klik, lalu memindahkannya mengikuti cursor pada bidang yang menghadap kamera. |

Semua brush kecuali Grab lewat `Brush::apply` dan dibatasi `kMaxDabMove` (0,2 × radius per dab) supaya pencarian normal basi tetap benar. Grab punya jalur sendiri di `Sculptor::beginGrab/grab`: daftar vertex, face, dan daun yang terpengaruh dihitung sekali saat klik, lalu tiap gerakan cursor hanya menulis posisi, normal, dan refit daun itu.

Undo menyimpan keadaan sebelum dan sesudah untuk daun yang berubah saja (posisi dan normal), dengan batas memori 1 GB. Entry yang topologinya sudah berubah (misalnya setelah remesh nanti) dilewati, bukan diterapkan ke mesh yang salah.

## Voxel remesh (`remesh/`)

Pipeline: mesh → `VoxelGrid` (signed distance di node grid) → `extractSurfaceNets` (quad) → `buildMesh` → BVH baru. Topologi awal tidak menentukan hasil.

- **Tanda dalam/luar** dihitung tepat di setiap node: sinar sepanjang tiap sumbu grid menghitung persilangan permukaan beserta arahnya (winding number). Node dianggap di dalam kalau minimal dua dari tiga sumbu setuju. Bagian yang saling tumpuk dan self-intersection jadi gabungan (union), arah face yang terbalik tidak berpengaruh, dan lubang kecil kalah suara.
- **Jarak** hanya disimpan di pita tipis (2 voxel) sekitar permukaan, dalam blok 8×8×8, jadi memori mengikuti luas permukaan. Tanda disimpan padat (1 byte per node).
- **Surface Nets** membuat satu vertex per patch permukaan di tiap sel dan satu quad per edge grid yang disilang permukaan, jadi hasilnya 100% quad dan tertutup. Sel yang dilewati dua lembar permukaan mendapat satu vertex per lembar. Face sel yang ambigu diputuskan dari nilai tengahnya dengan urutan penjumlahan tetap, jadi dua sel tetangga selalu sepakat dan setiap edge dipakai tepat dua quad (manifold).
- Remesh di aplikasi berjalan di worker thread. Undo menyimpan mesh lengkap sebelum dan sesudah (`TopologyUndo`), termasuk `topologyVersion`, sehingga undo stroke sebelum remesh tetap berlaku setelah remesh di-undo.

Keterbatasan saat ini: valence 4 baru sekitar 50% (khas Surface Nets pada permukaan melengkung). Optimasi valence, relaksasi, dan proyeksi ke permukaan asli ada di fase berikutnya.

## Threading (`core/Parallel.h`)

Engine hanya memakai satu primitif paralel, `parallelFor`. Implementasinya sekarang thread pool kecil sendiri; bisa diganti oneTBB tanpa mengubah pemanggil. Import OBJ dan build BVH berjalan di thread latar, lalu hasilnya ditambahkan ke scene di thread utama.

## Hasil benchmark Phase 0

Mesin cloud 4 thread, Release, quad sphere (`plegl_bench`):

| Vertex | Build BVH | Raycast rata-rata | Query bola | Refit penuh |
| ---: | ---: | ---: | ---: | ---: |
| 50K | 8 ms | 22 µs | 0,2 µs | 0,2 ms |
| 100K | 14 ms | 24 µs | 0,3 µs | 0,3 ms |
| 500K | 103 ms | 27 µs | 0,6 µs | 1,6 ms |
| 1M | 241 ms | 28 µs | 0,6 µs | 2,6 ms |

Import OBJ 1M vertex (parse + build topologi): sekitar 0,3 detik.

## Hasil benchmark stroke

`plegl_bench`, mesin cloud 4 thread, Release, quad sphere radius 1, 200 dab sepanjang busur (Grab: 200 gerakan cursor setelah satu klik). Waktu per dab termasuk snapshot undo, brush, normal, dan refit. Angka Phase 1:

| Vertex | Radius dab | Brush | Rata-rata | p95 |
| ---: | ---: | --- | ---: | ---: |
| 500K | 0,15 | Draw | 0,65 ms | 0,94 ms |
| 500K | 0,40 | Draw | 2,0 ms | 3,1 ms |
| 500K | 0,40 | Smooth | 1,4 ms | 2,0 ms |
| 1M | 0,15 | Draw | 0,80 ms | 1,1 ms |
| 1M | 0,40 | Draw | 2,9 ms | 3,9 ms |
| 1M | 0,40 | Smooth | 2,3 ms | 4,0 ms |

Phase 2, 1M vertex, p95 per dab:

| Brush | Radius 0,15 | Radius 0,40 |
| --- | ---: | ---: |
| Clay | 1,3 ms | 6,0 ms |
| Inflate | 1,3 ms | 4,5 ms |
| Flatten | 1,4 ms | 7,2 ms |
| Crease | 1,7 ms | 10,0 ms |
| Grab | 0,4 ms | 1,8 ms |

Target PRD (latensi brush di bawah 16 ms) masih terpenuhi pada 1M vertex dengan brush besar. Upload GPU belum termasuk angka ini; panel Performance di aplikasi menampilkan waktu dab dan latensi input-ke-frame secara langsung.

## Hasil benchmark voxel remesh (Phase 3)

`plegl_bench`, mesin cloud 4 thread, Release, quad sphere radius 1. Total termasuk tanda, jarak, Surface Nets, dan build half-edge:

| Vertex input | Voxel | Grid | Vertex output | Total | Memori grid |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 0,010 | 207³ | 188K | 0,17 s | 17 MB |
| 100K | 0,005 | 407³ | 754K | 0,75 s | 99 MB |
| 500K | 0,010 | 207³ | 188K | 0,42 s | 17 MB |
| 500K | 0,005 | 407³ | 754K | 1,1 s | 98 MB |
| 1M | 0,010 | 208³ | 188K | 0,69 s | 17 MB |
| 1M | 0,005 | 408³ | 754K | 1,6 s | 98 MB |

Target PRD: 100K < 1 s, 500K < 3 s, 1M < 5 s. Volume bola berubah kurang dari 2% pada voxel 0,04, dan dua bola yang tumpang tindih menjadi satu permukaan dengan volume gabungan yang benar (selisih di bawah 3%).
