# Arsitektur

Dokumen ini merangkum keputusan yang sudah diterapkan di kode. Rencana lengkap Phase 0 dan Phase 1 ada di dokumen rencana teknis proyek.

## Engine terpisah dari aplikasi

`plegl_engine` adalah library statis tanpa window, GPU, atau UI. Semua yang ada di dalamnya bisa diuji dan di-benchmark secara headless (`plegl_tests`, `plegl_bench`). Aplikasi (`src/app`) hanya lapisan tipis: input, kamera, renderer, dan panel.

## Mesh: half-edge berbasis index (`mesh/Mesh.h`)

- Topologi adalah array `int32`: `heNext`, `heTwin`, `heVert`, `heFace`, ditambah `vertHe` dan `faceHe`. Tidak ada pointer, jadi mesh murah disalin, diserialisasi, dan dikirim ke thread lain.
- Atribut disimpan sebagai struct-of-arrays (`positions`, `normals`, `mask`). Jalur panas brush hanya menyentuh array ini.
- Edge terbuka ditandai `heTwin == -1`, tanpa loop boundary eksplisit.
- Face boleh segitiga, quad, atau n-gon. Output remesh 100% quad.
- `buildMesh` membangun twin lewat adjacency per vertex (linear terhadap jumlah half-edge) dan melaporkan edge non-manifold, vertex non-manifold, dan face degenerate.
- `validate()` memeriksa semua invarian topologi. Dipakai di test dan nanti di build debug setiap selesai operasi topologi.

## BVH: node daun sebagai unit kerja (`spatial/Bvh.h`)

- Face dikelompokkan ke node daun berisi paling banyak 1.024 face (median split pada sumbu terpanjang).
- `Bvh::build` **mengurutkan ulang mesh** sehingga setiap daun memiliki rentang face dan rentang vertex yang bersambung. Vertex dimiliki oleh daun pertama yang memakainya.
- Karena itu satu daun bisa menjadi unit untuk raycast, query brush, upload GPU parsial (`glNamedBufferSubData` per rentang vertex), snapshot undo, dan pembagian kerja antar thread.
- Setelah vertex bergerak cukup `refitLeaves` (murah). Rebuild penuh hanya setelah topologi berubah.
- Raycast menelusuri node dari depan ke belakang dan berhenti begitu node berikutnya lebih jauh dari hit terbaik.

## Renderer (`src/app/Renderer.*`)

- Satu buffer posisi, satu buffer normal, dan satu buffer mask per objek, urutannya sama dengan mesh. Daun yang ditandai dirty diunggah per rentang, bukan seluruh buffer. Mask punya daftar dirty sendiri, jadi stroke mask tidak mengunggah ulang posisi.
- Indeks segitiga dan edge disimpan **per daun BVH** dalam blok sendiri (`render/LeafIndexBlocks.h`). Segitiga digambar dengan satu `glMultiDrawElementsIndirect` (satu perintah per rentang), edge dengan satu `glMultiDrawElements`. Saat mesh diam, blok-blok itu bersambung dan digambar sebagai satu rentang. Selama stroke dynamic topology hanya blok daun yang face-nya berubah yang ditulis ulang; blok yang tidak muat lagi pindah ke ruang kosong atau ke ujung buffer dengan sisa ruang.
- Buffer vertex dan index punya ruang tambahan (vertex: +25%, minimal 65.536) dan diperbesar dengan salinan di GPU (`glCopyNamedBufferSubData`) kalau stroke kehabisan ruang, tanpa upload ulang dari CPU.
- Wireframe memakai edge poligon asli (bukan diagonal segitiga), jadi quad tampil sebagai quad. Indeks edge hanya dijaga selama wireframe aktif dan dibangun ulang sekali saat wireframe dinyalakan.
- **Warna face set** dibaca fragment shader dari buffer storage berisi satu nilai per *slot* segitiga (index pertama segitiga / 3), dengan urutan yang sama dengan buffer indeks. Slot dihitung dari `gl_PrimitiveID` ditambah slot awal rentang yang sedang digambar. `gl_PrimitiveID` mulai dari 0 di setiap perintah indirect, dan slot awal itu dibaca sebagai atribut per instance dari field `firstIndex` di buffer perintah itu sendiri (perintah ke-i memakai `baseInstance` i). Karena itu setiap blok segitiga mulai dan memesan ruang dalam kelipatan 3 index. Mesh tanpa face set memakai buffer pengganti satu slot dan shader tidak membacanya. Set default (1) tidak diwarnai; set lain mendapat warna pastel dari rasio emas, jadi id berurutan tetap mudah dibedakan.
- **Face tersembunyi** tidak masuk buffer indeks. Edge di antara face terlihat dan tersembunyi digambar oleh sisi yang terlihat, jadi menyembunyikan face menandai daunnya dan daun tetangga di seberang edge-nya (`markVisibilityDirty`). Kalau lebih dari separuh daun berubah sekaligus (misalnya Reveal All), buffer indeks dibangun ulang utuh sekali dan tidak dipindah blok per blok.
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

Undo menyimpan keadaan sebelum dan sesudah untuk daun yang berubah saja (posisi dan normal, nilai mask untuk stroke dan operasi mask, atau nilai face set per face untuk stroke dan operasi face set), dengan batas memori 1 GB. Stroke yang tidak mengubah apa pun (misalnya seluruhnya di area ter-mask) tidak membuat entry, jadi riwayat redo tetap utuh. Entry yang topologinya sudah berubah (misalnya setelah remesh nanti) dilewati, bukan diterapkan ke mesh yang salah.

## Mask (`sculpt/MaskOps.h`, `Mesh::mask`)

- `Mesh::mask` berisi satu float per vertex di [0, 1], 1 berarti terlindungi penuh. Kosong berarti tidak ada yang di-mask, jadi mesh tanpa mask tidak membayar memori apa pun. Mask ikut diurutkan ulang oleh `reorder`, ikut `splitEdge` (interpolasi), `collapseEdge`/`collapseDiagonal` (nilai terbesar dipertahankan, supaya area terlindungi tidak bocor), dan `compact`.
- Setiap brush yang memindahkan vertex mengalikan gerakannya dengan (1 − mask). Grab mengalikan bobot tangkapnya saat klik. Vertex dengan mask 1 tidak bergerak sama sekali (bit-exact, dicek di test untuk semua brush).
- **Mask brush** (`MaskBrush`) menggeser nilai mask ke 1 (cat) atau 0 (hapus) sesuai falloff dan strength. Dengan Shift, `MaskSmoothBrush` meratakan mask ke rata-rata tetangga. Berbeda dengan Smooth biasa, vertex border dirata-rata dengan semua tetangganya, jadi tepi terbuka tidak terputus dari mask di sebelahnya. Stroke mask lewat jalur `Sculptor` yang sama tetapi hanya menyimpan nilai mask untuk undo, tidak menghitung normal, dan tidak me-refit BVH. Daun yang mask-nya ternyata tidak berubah dibuang dari entry undo, jadi mengecat area yang sudah penuh tidak menambah riwayat.
- **Operasi seluruh mesh** (`applyMaskOp`): Invert, Clear, Fill, Blur, dan Sharpen. Semuanya paralel per daun BVH, dan undo hanya menyimpan daun yang berubah. Blur menggeser tiap nilai setengah jalan ke rata-rata tetangganya; Sharpen mendorong nilai menjauhi hasil blur (unsharp mask) lalu dijepit ke [0, 1].
- **Tampilan**: area ter-mask digelapkan di shader (`showMask`, `maskOpacity`). Kalau lebih dari 64 daun berubah sekaligus (misalnya Invert), buffer mask diunggah utuh sekali, bukan per daun.
- **Penyimpanan**: chunk opsional `MASK` di `.psculpt` (lihat Project file). File tanpa chunk itu dibuka tanpa mask, dan build lama melewatinya, jadi versi format tidak berubah.
- **Remesh**: kalau sumbernya punya mask, setiap vertex hasil mengambil titik terdekat di mesh asli (BVH daun 8 face yang sudah dipakai untuk proyeksi) lalu menginterpolasi mask dari tiga sudut segitiganya. Biayanya sekitar 0,1 s untuk 157K vertex output.

## Face set (`sculpt/FaceSetOps.h`, `Mesh::faceSets`)

Face set adalah grup poligon berwarna, dipakai untuk memilih bagian model dengan cepat: membatasi brush, membuat mask, atau menyembunyikan bagian yang menghalangi.

- **Data**: `Mesh::faceSets` berisi satu `int32` per face. Nilai v berarti set |v|, dan v < 0 berarti face itu tersembunyi; 0 tidak pernah disimpan. Id paling besar `INT32_MAX − 1`, jadi id baru (id terbesar + 1) tidak pernah overflow; file dengan nilai di luar rentang itu ditolak. Kosong berarti semua face ada di set default (1) dan terlihat, jadi mesh tanpa face set tidak membayar memori apa pun. Face set ikut `reorder`, `splitFace` (kedua face hasil mewarisi set-nya), `compact`, relayout dynamic topology, dan undo per daun (rentang face daun itu).
- **Tersembunyi**: vertex terlihat kalau minimal satu face di sekitarnya terlihat. Brush, Grab, brush mask, dan operasi mask tidak mengubah vertex yang tidak terlihat. Raycast, picking, dan pencarian titik terdekat melewati face tersembunyi, jadi stroke dan cursor menembus bagian yang disembunyikan. Dynamic topology tidak menyentuh face tersembunyi.
- **Brush Face Set**: setiap stroke mengecat set baru (id terbesar + 1, sama di kedua sisi simetri). Dengan Ctrl, stroke memperluas set yang ada di bawah dab pertama. Face dicat kalau pusatnya ada di dalam dab dan bobot falloff-nya ≥ 1 − strength, jadi strength rendah mengecat inti yang lebih kecil. Face yang rata-rata mask sudutnya ≥ 0,5 dan face tersembunyi tidak dicat. Seperti stroke mask, stroke ini hanya menyimpan nilai face set untuk undo dan tidak menghitung normal.
- **Auto-mask** (untuk semua brush): *Limit to one set* membatasi stroke ke set di bawah dab pertama. Dengan simetri, sisi cermin memakai set di bawah dab cerminnya sendiri. *Keep set borders* mengunci vertex di batas antar set. Keduanya juga berlaku untuk Grab dan brush mask.
- **Operasi** (`applyFaceSetOp`): Face Set from Mask (face terlihat yang semua sudutnya ter-mask jadi set baru), Face Sets from Loose Parts (setiap bagian yang terpisah dapat set sendiri, mulai dari 2), Clear, Hide, Isolate (menyembunyikan semua di luar satu set; kalau sudah begitu, menampilkan semuanya lagi), Reveal All, dan Invert Visibility. Undo hanya menyimpan daun yang berubah. `maskFaceSet` me-mask semua vertex dari face terlihat di satu set.
- **Dynamic topology** menjaga batas set: kalau satu ujung edge ada di batas set, collapse mempertahankan ujung itu di tempatnya (dan menolak kalau mask lebih memilih ujung lain). Kalau kedua ujung di batas, collapse hanya boleh sepanjang edge batas itu sendiri. Batas tidak bergeser walaupun daerah di sekitarnya diperhalus dan diperkasar berulang kali (dicek di test). Auto-mask juga berlaku di sini: dengan *Limit to one set*, face dari set lain tidak dipecah atau digabung, dan dengan *Keep set borders* tidak ada titik batas yang digabung, jadi batasnya tetap sama titik demi titik.
- **Remesh**: setiap face hasil mengambil set dari face asli yang terdekat dengan pusatnya. Kalau normal keduanya berlawanan (misalnya bagian tipis), pencarian diulang dari titik sedikit di luar face.
- **Penyimpanan**: chunk opsional `FSET` di `.psculpt` (lihat Project file).

## Voxel remesh (`remesh/`)

Pipeline: mesh → `VoxelGrid` (signed distance di node grid) → `extractSurfaceNets` (quad) → `buildMesh` → BVH baru. Topologi awal tidak menentukan hasil.

- **Tanda dalam/luar** dihitung tepat di setiap node: sinar sepanjang tiap sumbu grid menghitung persilangan permukaan beserta arahnya (winding number). Node dianggap di dalam kalau minimal dua dari tiga sumbu setuju. Bagian yang saling tumpuk dan self-intersection jadi gabungan (union), arah face yang terbalik tidak berpengaruh, dan lubang kecil kalah suara.
- **Jarak** hanya disimpan di pita tipis (2 voxel) sekitar permukaan, dalam blok 8×8×8, jadi memori mengikuti luas permukaan. Tanda disimpan padat (1 byte per node).
- **Surface Nets** membuat satu vertex per patch permukaan di tiap sel dan satu quad per edge grid yang disilang permukaan, jadi hasilnya 100% quad dan tertutup. Sel yang dilewati dua lembar permukaan mendapat satu vertex per lembar. Face sel yang ambigu diputuskan dari nilai tengahnya dengan urutan penjumlahan tetap, jadi dua sel tetangga selalu sepakat dan setiap edge dipakai tepat dua quad (manifold).
- Remesh di aplikasi berjalan di worker thread. Selama itu undo dan redo ditahan, begitu juga stroke dan operasi mask pada objek yang di-remesh, karena hasilnya dibangun dari mesh saat remesh dimulai dan akan menimpa perubahan itu. Undo menyimpan mesh lengkap sebelum dan sesudah (`TopologyUndo`), termasuk `topologyVersion`, sehingga undo stroke sebelum remesh tetap berlaku setelah remesh di-undo.

Output Surface Nets mentah punya valence 4 sekitar 50% (khas pada permukaan melengkung). Itu diperbaiki oleh quad remesh di bawah.

## Operasi topologi (`mesh/MeshEdit.h`)

`MeshEditor` mengubah topologi secara lokal: `rotateEdge` (flip pada segitiga, rotasi edge pada quad), `splitEdge`, `splitFace`, `collapseEdge`, dan `collapseDiagonal` (menghapus satu quad dengan menggabungkan dua sudut yang berseberangan, sehingga semua face lain tetap quad). Elemen yang dihapus hanya ditandai mati, jadi index tetap stabil selama editing; `compact()` membuang yang mati dan menomori ulang sekali di akhir. Setiap operasi memeriksa link condition dan valence minimum, dan menolak (tanpa mengubah apa pun) kalau hasilnya tidak manifold. Operasi ini juga dipakai dynamic topology; untuk itu `MeshEditor` bisa diberi `EditObserver` yang diberi tahu setiap elemen sebelum ditulis.

## Dynamic topology (`sculpt/Dyntopo.h`, `spatial/LeafLayout.h`)

Dengan dynamic topology (Ctrl+D di Sculpt mode), setiap dab mengubah topologi di bawah brush sebelum brush berjalan: edge yang lebih panjang dari ukuran detail D dipecah, dan edge yang lebih pendek dari 0,4 D digabung. Detail bisa ditentukan dalam pixel layar (Relative, default 8 px, diubah ke satuan objek di titik kena, jadi zoom in menambah detail) atau sebagai panjang tetap dalam satuan objek (Constant). Tahan R lalu geser mouse untuk mengubahnya. D tidak pernah lebih halus dari radius/20, supaya satu dab tidak meledakkan jumlah face. Satu pass dibatasi 1.024 split, 1.024 collapse, dan 3 ms. Grab, Mask, dan Face Set tidak mengubah topologi.

**Split** mengikuti *longest-edge bisection*: segitiga hanya dipotong lewat edge terpanjangnya. Kalau segitiga di sebelah edge yang mau dipecah punya edge yang lebih panjang, edge itu dipecah dulu, walaupun di luar brush. Tanpa aturan ini, split berulang di samping edge panjang yang tidak terjangkau brush menghasilkan segitiga yang makin tipis tanpa pernah memendek (ditemukan oleh test). Segitiga yang edge terpanjangnya tidak boleh dipotong (berbatasan dengan face yang tidak disentuh) dibiarkan utuh, dan setiap edge yang menunggu segitiga itu dilepas untuk sisa pass, jadi edge lain di bawah brush tetap diproses. Titik baru diletakkan di tengah edge, lalu kedua face dipotong dari titik itu ke sudut seberangnya. Edge yang sama panjang diurutkan menurut index half-edge, jadi hasilnya sama di setiap compiler. Quad dipotong jadi dua segitiga (diagonal terpendek yang tidak terlipat) hanya kalau edge-nya perlu dipecah. Edge border juga dipecah di tengah, jadi bentuk border tetap.

**Collapse** menggabungkan kedua ujung edge di tengahnya (atau di ujung yang lebih ter-mask) dan ditolak kalau: ada face di sekitarnya yang tidak boleh diubah, valence hasilnya lebih dari 12, ada edge baru yang lebih panjang dari D (supaya split dan collapse tidak berosilasi), ada face yang mengecil hampir nol, ada face yang berputar lebih dari 60° dari normalnya sebelum collapse, ada face yang menyimpang lebih dari sekitar 78° dari normal permukaan di kedua ujung edge (acuan tetap, supaya face tidak miring sedikit demi sedikit lewat banyak collapse sampai terlipat), atau link condition `MeshEditor` gagal. Vertex border tidak pernah digabung.

**Tidak disentuh**: vertex dengan mask ≥ 0,5, vertex non-manifold (dicek sekali per versi topologi), setiap face yang punya sudut seperti itu, face dengan lebih dari 4 sudut, dan face tersembunyi. Vertex di batas face set tidak berpindah (lihat Face set).

**Layout selama stroke.** Edit dilakukan di tempat: elemen yang dihapus hanya ditandai mati, elemen baru masuk ke daun ekor BVH yang diperiksa raycast dan query bola lebih dulu. Index lama tidak pernah bergeser selama stroke, jadi brush, raycast, dan renderer tetap bekerja di tengah stroke. Ruang untuk elemen baru dipesan di awal stroke (paling sedikit 65.536 vertex) supaya tidak ada realokasi di dalam dab.

**Undo.** `ClaimRecorder` (sebuah `EditObserver`) menyalin satu daun utuh tepat sebelum tulisan pertama ke daun itu. Daun yang hanya bergerak tetap memakai snapshot posisi biasa. Di akhir stroke `consolidate` membangun ulang hanya daun yang tersentuh, dengan index daun yang stabil (daun yang diklaim dipakai lagi, lalu daun kosong, baru kemudian daun baru), dan menyalin sisanya dengan pergeseran. Hasilnya layout kanonik yang sama seperti mesh biasa, sehingga sisa engine tidak pernah melihat elemen mati. `DyntopoUndo` menyimpan sisi sebelum (daun yang diklaim dan snapshot posisi) ditambah pasangan referensi di batas wilayah; sisi sesudah baru disalin saat undo pertama. Undo dan redo adalah relayout yang bit-exact. Kalau ada perubahan yang tidak tercatat, `consolidate` mendeteksinya, membangun ulang semua daun, dan memutus riwayat sebelum stroke itu daripada menerapkannya ke mesh yang salah.

**GPU.** Setiap dab menandai daun yang segitiga atau edge-nya berubah (`SceneObject::topoDirtyLeaves`), dan renderer hanya menulis ulang blok indeks daun itu, ditambah rentang vertex dan mask daun yang berubah. Di akhir stroke versi topologi berubah, jadi mesh diunggah penuh sekali dengan indeks yang dibuat paralel per daun. Statistik mesh (persentase quad, valence 4) dihitung paralel satu frame kemudian, tidak pernah di tengah stroke.

## Multiresolution (`multires/`)

Objek bisa punya beberapa level subdivisi (paling banyak 8, termasuk base). Subdivide menambah satu level di atas dengan Catmull-Clark: setiap face dengan n sudut jadi n quad. Artis bisa membentuk bentuk besar di level rendah dan detail di level tinggi, dan perubahan di satu level menyebar ke level lain.

- **Data** (`Multires.h`): setiap level adalah `Mesh` dan `Bvh` lengkap dengan layout beku. BVH dibangun sekali, lalu hanya di-refit, dan setiap level punya `topologyVersion` sendiri seumur hidupnya. Level aktif ada di `SceneObject::mesh`/`bvh`, jadi brush, renderer, picking, mask, dan face set bekerja seperti biasa. Level lain diparkir di `SceneObject::multires`. Dynamic topology mati untuk objek yang punya level.
- **Penomoran kanonik**: urutan vertex dan face di memori mengikuti BVH, jadi bisa berbeda antar build. Karena itu setiap level juga menyimpan nomor kanonik yang hanya bergantung pada topologi base. Anak vertex memakai nomor vertex induknya, anak edge memakai nV + peringkat half-edge pemilik kanonik, dan anak face memakai nV + nE + nomor face induknya. Topologi level k+1 ditulis dalam bentuk tertutup dari level k: quad q(h) bersudut [vc(v), ec(h), fc(f), ec(prev h)], dan twin-nya dihitung langsung dari twin level k. Kernel subdivisi hanya mengumpulkan nilai (gather), dan titik edge dihitung sebagai (a+b)+(Fl+Fr), jadi hasilnya bit-exact untuk urutan memori dan jumlah thread mana pun (dicek di test).
- **Aturan vertex**: vertex yang terisolasi, non-manifold, atau berada di fan dengan terlalu sedikit face disematkan (Pinned) dan tidak bergerak. Vertex di border terbuka memakai aturan kurva border, dan sisanya memakai aturan Catmull-Clark biasa. Anak mewarisi aturan induknya.
- **Sinkronisasi** (`Propagate.h`): saat level diganti, `diffActive` membandingkan level aktif bit demi bit dengan referensi yang diambil waktu level itu diaktifkan. Hanya vertex yang berubah yang disebarkan. Ke atas, perubahan di-subdivisi ulang lewat ring Φ0/Φ1/Φ2, dan detail level tinggi dibawa dalam frame lokal (normal dan tangen), jadi detail ikut berputar saat bentuk dasarnya digeser. Ke bawah, perubahan diturunkan dengan restriksi tenda, misalnya untuk vertex biasa (4Δvc + 2ΣΔec + ΣΔfc) / (4 + 3n). Mask naik dengan interpolasi dan turun dengan injeksi. Face set naik apa adanya. Saat turun, induk mengambil id mayoritas anaknya (seri dimenangkan anak pertama), dan hanya tersembunyi kalau semua anaknya tersembunyi. Operasi yang menjangkau semua level menjaga aturan itu: Invert Visibility membalik level teratas, lalu setiap level di bawahnya mengikuti anaknya, dan Face Sets from Loose Parts memakai id yang belum dipakai di level mana pun.
- **Undo**: `MultiresUndo` mencatat Switch, Subdivide, DeleteHigher, DeleteLower, dan AllLevels (Clear/Fill/Invert mask, Clear/Reveal All/Invert Visibility face set di semua level), termasuk delta sinkronisasinya. Switch berturut-turut digabung jadi satu langkah. Stroke tetap bisa di-undo setelah pindah level lalu kembali, karena versi topologi level itu tidak berubah.
- **Data rusak**: posisi NaN atau tak hingga tidak boleh masuk ke level, karena akan menyebar ke level lain dan membuat file tidak bisa dibuka lagi. Import OBJ menolaknya, Subdivide menolak mesh yang memilikinya, dan saat membuka file level aktif juga diperiksa.
- **Di aplikasi**: Subdivide berjalan di worker thread. Selama itu objek tidak bisa diedit dan undo ditahan, sama seperti remesh. Remesh menghapus semua level, dan undo mengembalikannya. Pindah level membuat level yang datang diunggah ulang ke GPU. Kalau level pergi dan kembali dalam satu frame, semua daunnya ditandai kotor, jadi GPU tidak pernah memakai data lama.
- **Penyimpanan**: chunk `MRES` (lihat Project file).

## Sculpt layers (`mesh/LayerStack.h`, `sculpt/LayerOps.h`)

Layer menyimpan hasil sculpting secara non-destruktif. Setiap layer adalah offset per vertex dengan strength, dan bentuk yang terlihat adalah kompositnya. Detail bisa dikecilkan, dibalik, disembunyikan, atau dihapus kapan saja tanpa menyentuh layer lain.

- **Data**: `Mesh::layers` (`LayerStack`) berisi base (posisi tanpa layer), daftar layer (id, nama, strength −1000% sampai 1000%, visibilitas, offset per vertex), layer aktif (0 = base), dan epoch. Paling banyak 16 layer per objek (atau per level subdivisi) dan 1 GB per objek. Mesh tanpa layer punya stack yang persis kosong, jadi semua jalur lama tidak berubah.
- **Komposit di posisi**: `positions` selalu sama bit demi bit dengan base + Σ strength·offset dari layer yang terlihat, dan normal selalu sama dengan `computeNormals`. Komposit hanya dihitung oleh `composeVertex`: urutan tetap dari bawah ke atas, perkalian ±0 dilewati, dan tanpa FMA (`-ffp-contract=off` untuk GCC/Clang, `/fp:precise` di MSVC), jadi hasilnya sama di setiap platform. Karena itu brush, renderer, picking, remesh, dan export OBJ tidak perlu tahu soal layer. `validateLayers` memeriksa aturan ini di test.
- **Stroke** menulis ke layer aktif. Brush bekerja di posisi seperti biasa. Di akhir stroke, perubahan setiap vertex dibagi strength layer dan ditambahkan ke offset-nya, lalu posisi diambil lagi dari komposit (bisa beda pembulatan). Normal dan bounds di sekitar vertex yang bergeser karena pembulatan dihitung ulang paralel per daun. Di base, vertex yang tidak digeser layer mana pun mengambil hasil brush persis. Stroke ditolak di layer tersembunyi atau di bawah 5%. Brush Erase Layer memudarkan offset layer aktif ke nol, dan Smooth dengan "This layer only" menghaluskan offset layer, bukan bentuknya. Undo stroke ikut menyimpan offset per daun dan hanya berlaku pada state layer yang sama (kunci dari epoch, id, visibilitas, dan strength).
- **Operasi**: Add, Duplicate (salinan tersembunyi), Delete, Rename, Show/Hide, Solo, Show/Hide All, Strength, Invert, Merge Down, Apply, Apply All, dan Mask from Layer. Setiap operasi mengomposisi ulang hanya vertex yang dijangkau layer yang berubah (support), atau semua vertex dengan pass paralel kalau gabungan support lebih dari setengah mesh. Undo menyimpan array yang keluar dari stack secara utuh (dipindah, bukan disalin) atau sebagai patch nilai, lalu menghitung komposit lagi, jadi undo dan redo bit-exact.
- **Strength drag**: slider strength mengubah bentuk setiap frame. Normal ikut live sampai 600K vertex terjangkau (di atas itu diperbarui saat slider dilepas), refit BVH dilakukan saat dilepas, dan frame berikutnya dilewati kalau satu update lebih dari 12 ms. Satu drag adalah satu langkah undo.
- **Multiresolution**: setiap level punya stack sendiri. Sinkronisasi antar level membawa base dan offset: naik memutar offset dengan frame detail, turun menambahkan delta ke base. Merge Down dan Apply hanya mengubah pembulatan, jadi referensi level ikut digeser dan perubahan itu tidak dianggap edit. Level baru dari Subdivide mulai tanpa layer.
- **Interaksi lain**: dynamic topology tidak berjalan pada objek berlayer (stroke ditolak dengan alasan). Remesh memanggang layer ke mesh, dan undo mengembalikannya. Duplicate objek menyalin stack-nya.
- **Penyimpanan**: chunk `LAYR` (lihat Project file).

## Quad remesh (`remesh/QuadRemesh.h`)

Pipeline: `voxelRemesh` pada **2× target edge** → beberapa ronde (default 3) optimasi → subdivisi 1 level ke target edge → relaksasi.

Satu ronde optimasi:

1. **Collapse diagonal**: quad tipis (diagonal < 0,35 × edge) dan quad yang collapse-nya menurunkan ketidakteraturan valence (skor Σ(valence − 4)²) dihapus, misalnya pola 3-x-3-x yang ditinggalkan Surface Nets di bagian melengkung.
2. **Rotasi edge**: edge di antara dua quad diputar kalau enam vertex di sekitarnya jadi lebih dekat ke valence 4, selama tidak ada sudut quad yang terlipat.
3. **Drift** (default 5 pass): langkah 1 dan 2 berhenti di sekitar 95% karena sisa vertex tidak teratur berupa pasangan 3-5 yang tersebar (mirip dislokasi pada kisi). Memindahkan pasangan satu langkah tidak mengubah skor, dan pasangan baru bisa saling hilang kalau bertemu. Pass drift mengambil langkah yang skornya tetap, tetapi memilih yang mendekatkan vertex tidak teratur ke vertex tidak teratur lain dalam radius 3 ring. Pass perbaikan sesudahnya menghapus pasangan yang sudah bertemu.
4. **Relaksasi**: vertex bergeser ke rata-rata tetangganya sepanjang bidang singgung. Langkah terakhir memproyeksikannya ke mesh asli (`Bvh::closestPoint`, daun kecil 8 face) dengan cek arah normal supaya tidak menempel ke sisi seberang bagian tipis.

**Subdivisi** memecah setiap quad menjadi empat. Vertex lama mempertahankan valence-nya dan semua vertex baru bervalence 4, jadi porsi vertex tidak teratur turun sekitar empat kali. Ini yang membawa hasil dari ~98% ke ~99,5%. Harga yang dibayar: volume yang lebih tipis dari 2× edge bisa hilang pada langkah voxel, dan sudut tajam sedikit lebih bulat. Permukaannya tetap dipasang ke mesh asli pada target edge. `subdivisions = 0` memakai pipeline lama tanpa subdivisi.

Valence disimpan di cache dan diperbarui per operasi, sehingga pass topologi tetap linear. Hasil tetap tertutup, manifold, dan 100% quad. `measureQuality` memberi rasio valence 4, koefisien variasi panjang edge, dan galat permukaan (jarak pusat face ke mesh asli).

Di aplikasi, toggle **Optimize quads** (default aktif) memilih antara quad remesh dan voxel remesh biasa.

## Project file (`io/Project.h`)

Format `.psculpt` adalah container biner kecil: magic `PSCULPT\x1A`, versi, lalu chunk bertag (`u32 tag`, `u64 ukuran`, isi) dan chunk `END ` berisi CRC-32 dari semua byte sebelumnya.

- `OBJS` menyimpan tiap objek: nama, transform, visibilitas, posisi vertex (float, bit-exact), ukuran face, dan index face. Topologi half-edge dibangun ulang saat dibuka, dan BVH dibangun di worker thread.
- `MASK` (opsional, hanya kalau ada objek yang punya mask) menyimpan per objek: index objek di `OBJS`, jumlah vertex, encoding (0 = float32), lalu nilai mask. File dengan chunk `MASK` yang index objeknya salah, jumlah vertex-nya tidak cocok, encoding-nya tidak dikenal, atau terpotong ditolak ("The mask data is damaged."), sama seperti chunk lain yang rusak. Encoding baru nanti butuh versi format baru atau tag chunk baru. Nilai di luar [0, 1] atau NaN dijepit.
- `FSET` (opsional, hanya kalau ada objek dengan face set atau face tersembunyi) menyimpan per objek: index objek, jumlah face, encoding (0 = int32), lalu nilai face set dengan tanda tersembunyi. Jumlah face yang tidak cocok dengan mesh yang dibangun, encoding yang tidak dikenal, nilai 0, atau data terpotong ditolak dengan pesan bahwa data face set rusak atau terpotong.
- `MRES` (opsional, hanya kalau ada objek dengan level subdivisi) menyimpan per objek: topologi base dalam urutan kanonik (ukuran face, sudut, twin, dan awal fan setiap vertex), nilai setiap level yang tidak aktif, dan edit level aktif yang belum disebarkan. Untuk objek seperti itu, `OBJS`, `MASK`, dan `FSET` berisi level aktif dalam urutan kanonik, jadi build lama yang melewati `MRES` tetap membuka level yang sedang dipahat. Saat dibuka, semua level dibangun ulang dengan subdivisi dari base, lalu jumlah elemen dan poligon level aktif harus cocok persis dengan `OBJS`. Kalau tidak cocok, file ditolak ("The subdivision level data is damaged"). Hasilnya bit-exact, termasuk edit yang belum disebarkan (dicek di test).
- `LAYR` (opsional, sesudah `MRES`, hanya kalau ada objek berlayer) menyimpan per stack (objek, atau objek dan level, berurutan): layer aktif, next id, base, lalu per layer id, strength, visibilitas, nama UTF-8, dan offset. Setiap array disimpan padat (code 0) atau jarang (code 1: index naik dan nilai, nol di tempat lain; code 2 khusus base: komposit di tempat lain), mana yang lebih kecil. `OBJS` dan `MRES` tetap menyimpan bentuk komposit, jadi build lama tanpa layer membuka bentuk yang sedang dipahat (layer hilang kalau file itu disimpan ulang dari build lama). Saat dibuka, komposit dihitung ulang dan dibandingkan dengan bentuk tersimpan: sama persis dipakai apa adanya, beda pembulatan (sampai 1e-5 relatif) dihitung ulang, dan beda yang lebih jauh membuang layer objek itu dengan peringatan. Chunk `LAYR` yang rusak membuang semua layer dengan peringatan di status bar, tetapi project tetap terbuka dengan bentuknya. Encoding yang lebih baru juga dibuang dengan peringatan.
- `SETT` menyimpan setelan aplikasi sebagai baris `key value`: brush, radius, strength per brush, falloff, tekanan pen, simetri, kamera, viewport, remesh, mode, dan objek terpilih. Engine menyimpannya apa adanya, jadi setelan baru tidak perlu mengubah format. Kunci yang tidak ada memakai nilai default.
- Pembaca melewati chunk yang tidak dikenal (kompatibel ke depan) dan menolak file yang terpotong, rusak (checksum salah), atau dari versi format yang lebih baru, dengan pesan yang jelas. Setiap jumlah dan index diperiksa sebelum dipakai.
- Penyimpanan bersifat atomik: data ditulis ke `<nama>.tmp` lalu di-rename, jadi crash saat menyimpan tidak merusak project yang ada. Penyimpanan dibagi dua: `draftProject` menyalin data scene di thread utama (layer disalin sebagai daftar jarang atau array utuh), lalu `finishProject` (encoding `LAYR` dan CRC-32 slicing-by-8) dan penulisan file berjalan di worker, jadi scene boleh berubah selama itu.

**Autosave dan pemulihan.** Setiap 5 menit (bisa diubah di menu File, 0 = mati), kalau scene berubah sejak penyimpanan atau autosave terakhir, aplikasi menulis `autosave.psculpt` ke folder data pengguna (`SDL_GetPrefPath`). File `session.lock` dibuat saat start dan dihapus bersama autosave saat aplikasi ditutup normal. Kalau saat start lock masih ada, sesi sebelumnya berakhir tidak normal, dan aplikasi menawarkan **Recover Previous Session?**. Hasil pemulihan kembali ke path project aslinya tetapi ditandai belum disimpan.

**Perubahan yang belum disimpan** dilacak dengan sidik jari murah dari daftar objek (id, nama, visibilitas, transform, `topologyVersion`) ditambah penghitung edit (stroke, undo, redo), tanpa meng-hash data vertex. Judul jendela menampilkan `*`, dan New, Open, serta Quit menanyakan **Save changes?** dulu.

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

Phase 6, mask brush pada 1M vertex: p95 di bawah 1,6 ms untuk semua radius (tanpa normal dan refit). Operasi mask seluruh mesh, dijalankan di thread utama:

| Vertex | Invert | Blur ×2 | Sharpen ×2 | Clear | Undo |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500K | 1,2 ms | 17 ms | 16 ms | 0,8 ms | 3,8 MB |
| 1M | 2,7 ms | 37 ms | 37 ms | 1,5 ms | 7,6 MB |

Phase 6b, Draw dengan dynamic topology (detail = radius/12, jadi mesh 500K dihaluskan di dab kecil dan dikasarkan di dab besar). "Topologi" adalah bagian waktu dab untuk split dan collapse. "Akhir stroke" adalah `consolidate` sekali setelah dab terakhir:

| Vertex | Radius dab | Rata-rata | p95 | Topologi | Split | Collapse | Akhir stroke | Undo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500K | 0,05 | 1,1 ms | 1,6 ms | 0,4 ms | 19.521 | 0 | 33 ms | 1,7 MB |
| 500K | 0,15 | 1,4 ms | 2,9 ms | 0,4 ms | 0 | 6.063 | 35 ms | 3,3 MB |
| 500K | 0,40 | 5,1 ms | 8,3 ms | 2,5 ms | 2.652 | 42.385 | 32 ms | 7,0 MB |
| 1M | 0,15 | 2,2 ms | 4,2 ms | 1,0 ms | 249 | 29.773 | 91 ms | 5,9 MB |
| 1M | 0,40 | 10,7 ms | 13,9 ms | 5,2 ms | 0 | 5.204 | 71 ms | 6,1 MB |

Mesin cloud ini dipakai bersama, jadi angkanya bergeser antar-run: pada 1M dengan radius 0,40, p95 berkisar 12,5 sampai 16 ms, tepat di batas target PRD. Di brush sebesar itu sebagian besar waktu topologi habis untuk mengumpulkan edge di dalam bola, sehingga batas 3 ms per pass sudah tercapai sebelum banyak collapse sempat jalan; itu tempat optimasi berikutnya.

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

## Hasil quad remesh (Phase 4)

`plegl_bench`, mesin dan mesh sama. Total termasuk voxel remesh:

| Vertex input | Target edge | Vertex output | Voxel | Optimasi | Total | Valence 4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 0,010 | 157K | 0,11 s | 0,27 s | 0,39 s | 99,5% |
| 100K | 0,005 | 628K | 0,23 s | 1,0 s | 1,3 s | 99,6% |
| 500K | 0,010 | 157K | 0,36 s | 0,46 s | 0,84 s | 99,5% |
| 500K | 0,005 | 628K | 0,54 s | 1,3 s | 1,8 s | 99,6% |
| 1M | 0,010 | 157K | 0,66 s | 0,67 s | 1,3 s | 99,5% |
| 1M | 0,005 | 628K | 0,84 s | 1,4 s | 2,2 s | 99,6% |

Kualitas dibanding voxel remesh biasa pada target edge yang sama (tes `quad remesh reaches 98% ...`, galat dalam satuan panjang edge rata-rata). Mesh tes ini kecil (7K sampai 13K vertex); makin besar mesh, makin tinggi rasio valence 4:

| Mesh | Valence 4 | CV panjang edge | Galat rata-rata |
| --- | --- | --- | --- |
| Quad sphere | 51% → 98,6% | 0,21 → 0,18 | 0,020 → 0,009 |
| UV sphere | 48% → 98,7% | 0,22 → 0,18 | 0,025 → 0,010 |
| Cube | 99,9% → 99,9% | 0,11 → 0,06 | 0,012 → 0,017 |
| Dua bola tumpang tindih | 52% → 99,1% | 0,22 → 0,17 | 0,022 → 0,012 |

Volume berubah kurang dari 1% dibanding hasil voxel. Rusuk kubus lebih bulat daripada voxel remesh biasa karena layout dibangun di 2× edge; penjagaan fitur tajam belum ada.

## Hasil multiresolution (Phase 6d)

`plegl_bench`, mesin yang sama (4 worker thread). Base quad sphere 13.824 face, brush Draw radius 0,15. "Ganti level" diukur tanpa edit yang menunggu. "Sinkron turun" adalah pindah satu level ke bawah setelah stroke 200 dab di level teratas, termasuk menyebarkan stroke itu ke semua level:

| Face | Level | Subdivide (worker) | Ganti level | Dab rata-rata | Sinkron turun | Simpan | Buka | Memori level lain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 55K | 1 | 14 ms | 0,1 ms | 0,16 ms | 0,8 ms | 8 ms | 34 ms | 3 MB |
| 221K | 2 | 64 ms | 0,3 ms | 0,36 ms | 2,7 ms | 29 ms | 0,18 s | 16 MB |
| 885K | 3 | 0,39 s | 1,7 ms | 0,81 ms | 9,0 ms | 0,11 s | 0,86 s | 67 MB |
| 3,5M | 4 | 1,8 s | 5,6 ms | 1,3 ms | 33 ms | 0,57 s | 4,0 s | 269 MB |

Bagian subdivide di thread utama di bawah 7 ms. Menyimpan objek 3,5M face dengan level hanya sekitar 0,12 s lebih lama daripada mesh biasa dengan jumlah face yang sama (0,42 s). Membuka file berjalan di worker thread.

## Hasil sculpt layer (Phase 6e)

`plegl_bench`, mesin yang sama (4 worker thread). Quad sphere dengan empat layer berisi stroke Draw, ditambah satu layer padat (semua vertex bergeser) untuk operasi utuh. Waktu dalam ms:

| Vertex | Dab Erase Layer | Dab Smooth layer | Akhir stroke (vertex bergeser) | Drag 10% | Drag padat | Lepas drag | Show/Hide padat | Merge padat | Apply | Apply All | Simpan, tambahan thread utama | Simpan, worker | Buka |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 0,21 | 0,29 | 0,9 (25K) | 0,4 | 1,8 | 0,3 | 2,1 | 3,5 | 0,4 | 0,01 | 1,4 | 3,4 | 52 |
| 500K | 0,5 | 0,8 | 3,3 (127K) | 1,5–2,0 | 8,7–9,3 | 1,3 | 8–12 | 15 | 1,3 | 0,01 | 4–5 | 19 | 260–300 |
| 1M | 0,8 | 1,2 | 6 (252K) | 5–7 | 5–7 | 12–17 | 18–25 | 32 | 2,3 | 0,01 | 8–10 | 36–43 | 550–650 |

- Dab di layer sama dengan dab biasa; akhir stroke di layer menambah sekitar 3 ms untuk 127K vertex di 500K (mengomposisi ulang, lalu normal dan bounds di sekitar vertex yang berbeda pembulatan).
- Drag padat di 500K menghitung normal live setiap frame, jadi sekitar 9 ms per frame (frame berikutnya dilewati kalau lebih dari 12 ms). Di 1M normal ditunda sampai slider dilepas, jadi frame lebih murah tetapi pelepasan lebih mahal.
- Operasi utuh pada layer padat didominasi oleh `computeNormals` (sekitar 7 ms di 500K dengan 4 thread). Rentang angka menunjukkan variasi antar run di mesin ini.
- Simpan: tambahan di thread utama adalah menyalin layer (daftar jarang atau array utuh); encoding dan CRC (sekitar 1,5–1,9 GB/s) berjalan di worker. Buka mencakup parse, CRC, dan build BVH untuk seluruh file (file 500K dengan layer sekitar 32 MB, tanpa layer 15 MB).
