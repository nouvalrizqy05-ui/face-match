"""
app.py
======
Aplikasi Streamlit: Face Comparison System
Berbasis ArcFace (512-dim) + PCA (95-dim) — Hybrid Scoring

Kelompok: Nouval · Tirta · Farritz
Mata Kuliah: Aljabar Linier | UNNES 2025

Jalankan:
    streamlit run app.py

Deploy ke Streamlit Cloud:
    1. Push ke GitHub (pastikan model/model_wajah.pkl ada)
    2. Buka share.streamlit.io → New app → pilih repo

Struktur:
    Tab 1 — Bandingkan Dua Foto (fokus utama, masa kecil vs dewasa)
    Tab 2 — Visualisasi PCA (untuk presentasi dosen)
    Tab 3 — Panduan & Penjelasan Matematis
    Tab 4 — Uji Publik (upload ZIP database sendiri)
"""

import os
import sys
import warnings
import pickle
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.scorer import HybridScorer, THRESHOLD_MATCH, WEIGHT_ONNX, WEIGHT_PCA
from src.visualizer import (
    plot_hybrid_score,
    plot_pca_variance,
    plot_embedding_comparison,
    plot_face_crops,
)

# ------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Face Comparison — ArcFace + PCA",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/model_wajah.pkl"

# ------------------------------------------------------------------
# HELPER: konversi PIL → OpenCV BGR
# ------------------------------------------------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------
# LOAD MODEL & EMBEDDER (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model PCA...")
def load_scorer() -> HybridScorer | None:
    try:
        return HybridScorer(MODEL_PATH)
    except FileNotFoundError as e:
        return None

@st.cache_resource(show_spinner="Memuat ArcFace embedder...")
def load_embedder():
    from src.embedder import FaceEmbedder
    return FaceEmbedder(detector_backend="opencv")


def get_embedding_safe(embedder, image_bgr: np.ndarray):
    """
    Wrapper dengan error handling yang informatif.
    Returns (embedding, crop) atau (None, None) dengan pesan error di st.
    """
    try:
        emb, crop = embedder.get_embedding(image_bgr, enforce_detection=True)
        if emb is None:
            st.warning(
                "⚠️ Wajah tidak terdeteksi.\n\n"
                "**Tips:**\n"
                "- Pastikan wajah menghadap depan (frontal)\n"
                "- Pencahayaan cukup dan merata\n"
                "- Tidak ada obyek yang menutupi wajah\n"
                "- Resolusi foto minimal 100×100 px"
            )
            return None, None
        return emb, crop
    except RuntimeError as e:
        if "DeepFace tidak tersedia" in str(e):
            st.error(
                "❌ **ArcFace belum terinstall.**\n\n"
                "Jalankan di terminal:\n"
                "```\npip install deepface tf-keras\n```"
            )
        else:
            st.error(f"❌ Error saat ekstraksi embedding: {e}")
        return None, None


# ------------------------------------------------------------------
# KOMPONEN UI: Hasil Diagnostik Lengkap
# ------------------------------------------------------------------
def render_result(result, emb1, emb2, crop1, crop2, scorer: HybridScorer,
                  label1="Foto 1", label2="Foto 2"):
    """Render seluruh blok hasil: keputusan, metrik, visualisasi."""

    # --- Keputusan utama ---
    if result.is_match:
        st.success(f"## ✅ WAJAH MIRIP — Kemungkinan orang yang **sama**")
    else:
        st.error(f"## ❌ WAJAH BERBEDA — Kemungkinan orang yang **berbeda**")

    # --- Progress bar hybrid score ---
    pct = max(0.0, result.hybrid_score)
    st.markdown(f"**Hybrid Score: {pct*100:.1f}%**")
    st.progress(min(1.0, pct))

    # --- 4 metric cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Hybrid Score",
        f"{result.hybrid_score:.4f}",
        delta=f"{'≥' if result.is_match else '<'} {THRESHOLD_MATCH} ({'MIRIP' if result.is_match else 'BEDA'})",
    )
    c2.metric("ArcFace (512-dim)", f"{result.sim_onnx:.4f}", delta=f"bobot ×{WEIGHT_ONNX:.0%}")
    c3.metric("PCA (95-dim)", f"{result.sim_pca:.4f}", delta=f"bobot ×{WEIGHT_PCA:.0%}")
    c4.metric(f"{result.confidence_icon} Keyakinan", result.confidence)

    st.divider()

    # --- Visualisasi ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Wajah yang Diproses**")
        fig_crops = plot_face_crops(crop1, crop2, result, label1, label2)
        st.pyplot(fig_crops, use_container_width=True)

    with col_b:
        st.markdown("**Dekomposisi Hybrid Score**")
        fig_bar = plot_hybrid_score(result)
        st.pyplot(fig_bar, use_container_width=True)

    # --- Posisi di ruang PCA 2D ---
    if emb1 is not None and emb2 is not None:
        with st.expander("📐 Lihat posisi di ruang PCA 2D (PC1 vs PC2)"):
            fig_pca2d = plot_embedding_comparison(
                emb1, emb2, scorer._pca, label1, label2
            )
            st.pyplot(fig_pca2d, use_container_width=True)
            st.caption(
                "Setiap titik = representasi wajah di 2 komponen PCA pertama. "
                "Semakin dekat posisinya, semakin mirip wajahnya."
            )

    # --- Penjelasan teknis ---
    with st.expander("ℹ️ Cara membaca hasil ini"):
        st.markdown(f"""
**Formula Hybrid Score:**
```
hybrid = (sim_arcface × {WEIGHT_ONNX}) + (sim_pca × {WEIGHT_PCA})
       = ({result.sim_onnx:.4f} × {WEIGHT_ONNX}) + ({result.sim_pca:.4f} × {WEIGHT_PCA})
       = {result.sim_onnx*WEIGHT_ONNX:.4f} + {result.sim_pca*WEIGHT_PCA:.4f}
       = {result.hybrid_score:.4f}
```

**Mengapa ArcFace 70% + PCA 30%?**
- **ArcFace** (70%): menangkap geometri wajah di ruang 512-dim — landmark jarak mata, 
  hidung, mulut. Ini yang membuat sistem bisa mengenali orang yang sama meski beda usia.
- **PCA** (30%): menangkap pola variasi utama dari 82 subjek FGNET lintas usia.
  PCA "belajar" bahwa variasi usia adalah variasi yang wajar, bukan perbedaan identitas.

**Threshold {THRESHOLD_MATCH}**: ditetapkan dari evaluasi BioID pada dataset FGNET — 
nilai di bawah ini menunjukkan probabilitas rendah bahwa kedua foto adalah orang yang sama.

**Tingkat Keyakinan:**
| Skor | Level |
|---|---|
| ≥ 0.70 | 🟢 Sangat Tinggi |
| 0.60–0.70 | 🟡 Tinggi |
| 0.50–0.60 | 🟠 Sedang |
| < 0.50 | 🔴 Rendah (berbeda) |
        """)


# ------------------------------------------------------------------
# LOAD RESOURCES
# ------------------------------------------------------------------
scorer  = load_scorer()
embedder = load_embedder()

# ------------------------------------------------------------------
# SIDEBAR DITIADAKAN
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("👤 Face Comparison System")

if scorer is None:
    st.error(
        "⚠️ **Model belum ditemukan.**\n\n"
        "Pastikan `model/model_wajah.pkl` ada di folder proyek. "
        "File ini sudah tersedia dari repo referensi BioID. "
        "Atau jalankan `python train.py` untuk melatih dari FGNET."
    )
    st.stop()

if not embedder.is_available():
    st.error(
        "⚠️ **DeepFace / ArcFace belum terinstall.**\n\n"
        "```bash\npip install deepface tf-keras\n```"
    )
    st.stop()

# Info model ditiadakan di frontend
# ------------------------------------------------------------------
# NAVIGASI TAB
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Bandingkan Dua Wajah",
    "📊 Visualisasi PCA",
    "📋 Penjelasan Matematis",
    "🌐 Uji Publik",
])

# ================================================================
# TAB 1 — BANDINGKAN DUA WAJAH (FOKUS UTAMA)
# ================================================================
with tab1:
    st.subheader("Bandingkan Dua Foto Wajah")
    st.info(
        "**Cara kerja:** Setiap foto diekstrak menjadi vektor 512-dimensi oleh ArcFace, "
        "yang menangkap geometri wajah (jarak mata, hidung, mulut). "
        "Vektor ini kemudian direduksi via PCA dan digabung menjadi satu hybrid score.\n\n"
        "**Cocok untuk:** foto masa kecil vs sekarang, foto dari sudut berbeda, "
        "foto dengan pencahayaan berbeda."
    )

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader(
            "📁 Foto #1 (mis. masa kecil)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="tab1_f1"
        )
        label1 = st.text_input("Label foto #1", value="Foto Masa Kecil", key="lbl1")
        if file1:
            st.image(file1, caption=label1, use_container_width=True)

    with col2:
        file2 = st.file_uploader(
            "📁 Foto #2 (mis. sekarang)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="tab1_f2"
        )
        label2 = st.text_input("Label foto #2", value="Foto Sekarang", key="lbl2")
        if file2:
            st.image(file2, caption=label2, use_container_width=True)

    if file1 and file2:
        if st.button("🔍 Bandingkan Sekarang", type="primary",
                     use_container_width=True, key="btn_compare"):

            with st.spinner("Mengekstrak ArcFace embedding dari kedua foto..."):
                img1_bgr = pil_to_bgr(Image.open(file1))
                img2_bgr = pil_to_bgr(Image.open(file2))
                emb1, crop1 = get_embedding_safe(embedder, img1_bgr)
                emb2, crop2 = get_embedding_safe(embedder, img2_bgr)

            if emb1 is not None and emb2 is not None:
                result = scorer.score(emb1, emb2)
                st.divider()
                render_result(result, emb1, emb2, crop1, crop2, scorer, label1, label2)

    elif not file1 or not file2:
        st.caption("Upload kedua foto untuk mulai perbandingan.")


# ================================================================
# TAB 2 — VISUALISASI PCA
# ================================================================
with tab2:
    st.subheader("Visualisasi Model PCA")
    st.caption(
        "PCA dilatih dari ~955 embedding ArcFace (512-dim) yang diekstrak dari "
        "dataset FGNET — 82 subjek dengan foto dari usia 0 hingga 69 tahun. "
        "Tujuan: menemukan komponen utama variasi wajah yang *tidak bergantung pada usia*."
    )

    pca = scorer._pca

    st.markdown("### 📈 Explained Variance")
    fig_var = plot_pca_variance(pca)
    st.pyplot(fig_var, use_container_width=True)

    st.markdown(
        f"Dengan **{pca.n_components_} komponen PCA** (dari 512 dimensi ArcFace), "
        f"model mempertahankan **{pca.explained_variance_ratio_.sum()*100:.1f}%** informasi. "
        f"PC1 saja menjelaskan **{pca.explained_variance_ratio_[0]*100:.1f}%** variansi total — "
        "kemungkinan merepresentasikan perbedaan usia atau jenis kelamin secara global."
    )

    st.divider()
    st.markdown("### 🗺️ Posisi Dua Wajah di Ruang PCA 2D")
    st.info("Upload dua foto di Tab 1 dan klik Bandingkan — lalu buka expander 'Lihat posisi di ruang PCA 2D'.")

    st.divider()
    st.markdown("### 🧮 Matriks Komponen PCA")
    with st.expander("Lihat 5 komponen pertama (PC1–PC5)"):
        for i in range(min(5, pca.n_components_)):
            comp = pca.components_[i]
            st.markdown(f"**PC{i+1}** — variance: `{pca.explained_variance_ratio_[i]*100:.2f}%`")
            st.markdown(
                f"Min: `{comp.min():.4f}` | Max: `{comp.max():.4f}` | "
                f"Norm: `{np.linalg.norm(comp):.4f}`"
            )
        st.caption(
            "Setiap baris adalah satu principal component — "
            "vektor 512-dim yang merepresentasikan 'arah variasi' dalam ruang embedding ArcFace."
        )


# ================================================================
# TAB 3 — PENJELASAN MATEMATIS
# ================================================================
with tab3:
    st.subheader("Penjelasan Matematis — Pipeline Lengkap")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🔬 Alur Training (Offline)")
        st.markdown("""
**Langkah 1 — Kumpulkan Dataset**
```
FGNET/images/
├── 001A02.JPG  ← subjek 001, usia 2 th
├── 001A43.JPG  ← subjek 001, usia 43 th
├── 002A03.JPG  ← subjek 002, usia 3 th
└── ...
82 subjek × ~12 foto = 955 gambar
```

**Langkah 2 — Ekstraksi ArcFace Embedding**
```
Setiap foto → DeepFace.represent(model="ArcFace")
            → vektor 512-dim, L2-normalized
```
ArcFace dilatih dengan Additive Angular Margin Loss,
sehingga wajah orang yang sama selalu berdekatan di
hipersfer, terlepas dari usia, ekspresi, atau sudut.

**Langkah 3 — Bentuk Matriks X**
```
X ∈ ℝ^(955 × 512)
Setiap baris = embedding satu foto
```

**Langkah 4 — Fit PCA**
```python
pca = PCA(n_components=0.95)
pca.fit(X)
# Xc = X − mean(X)   ← centering otomatis
# Xc = U Σ Vᵀ        ← SVD
# Pilih k komponen terkecil yang capai 95% variance
```
Hasil: `pca.components_` berukuran (95, 512)

**Langkah 5 — Simpan Model**
```python
pickle.dump({"pca_model": pca}, open("model_wajah.pkl","wb"))
```
        """)

    with col_b:
        st.markdown("### ⚡ Alur Inference (Real-time)")
        st.markdown(f"""
**Langkah 1 — Input 2 Foto**

**Langkah 2 — Deteksi & Alignment**
```
OpenCV Haar Cascade → bounding box wajah
DeepFace → face alignment (eye-nose-mouth keypoints)
```

**Langkah 3 — ArcFace Embedding**
```
foto₁ → ArcFace → emb₁ ∈ ℝ^512  (L2-normalized)
foto₂ → ArcFace → emb₂ ∈ ℝ^512  (L2-normalized)
```

**Langkah 4 — Hybrid Scoring**
```python
# Cosine similarity di ruang ArcFace 512-dim
sim_onnx = dot(emb₁, emb₂)   # karena sudah L2-norm

# Proyeksi ke ruang PCA 95-dim
p₁ = pca.transform([emb₁])[0]   # Xc · Vk
p₂ = pca.transform([emb₂])[0]

# Cosine similarity di ruang PCA
sim_pca = dot(p₁/‖p₁‖, p₂/‖p₂‖)

# Hybrid score
hybrid = {WEIGHT_ONNX} × sim_onnx + {WEIGHT_PCA} × sim_pca
```

**Langkah 5 — Keputusan**
```python
is_match = hybrid ≥ {THRESHOLD_MATCH}
```

**Mengapa PCA di atas ArcFace embedding?**

ArcFace sudah sangat baik dalam 512-dim. PCA menambahkan
"filter" yang telah belajar dari distribusi wajah lintas
usia (FGNET) — memperkuat dimensi yang relevan untuk
identitas dan meredam noise variasi usia/kondisi foto.
        """)

    st.divider()
    pca = scorer._pca
    st.markdown("### 📊 Statistik Model Aktif")
    rows = [
        ("Embedding model", "ArcFace (DeepFace)"),
        ("Input dimensi", f"{pca.components_.shape[1]} (ArcFace output)"),
        ("Output dimensi (k)", f"{pca.n_components_}"),
        ("Variance retained", f"{pca.explained_variance_ratio_.sum()*100:.2f}%"),
        ("PC1 variance", f"{pca.explained_variance_ratio_[0]*100:.2f}%"),
        ("Threshold keputusan", f"hybrid_score ≥ {THRESHOLD_MATCH}"),
        ("Bobot ArcFace", f"{WEIGHT_ONNX:.0%}"),
        ("Bobot PCA", f"{WEIGHT_PCA:.0%}"),
        ("Training dataset", "FGNET Aging Database (82 subjek, ~955 foto)"),
        ("SVD solver", "auto (sklearn memilih otomatis)"),
    ]
    for key, val in rows:
        st.markdown(f"- **{key}:** {val}")


# ================================================================
# TAB 4 — UJI PUBLIK
# ================================================================
with tab4:
    st.subheader("🌐 Uji Publik — Upload Database Wajah Sendiri")
    st.caption(
        "Siapapun bisa mencoba sistem ini dengan database wajah mereka sendiri. "
        "Upload ZIP → sistem ekstrak ArcFace embedding → fit PCA sementara → bandingkan dua foto.\n\n"
        "**Catatan:** Model di sesi ini hanya berlaku sementara dan tidak menimpa model kelompok."
    )

    st.warning(
        "⚠️ **Keterbatasan Uji Publik:** Karena PCA dilatih ulang dari database kecil yang kamu upload "
        "(bukan dari 955 foto FGNET), akurasi mungkin lebih rendah dari Tab 1. "
        "Untuk hasil terbaik, gunakan minimal 20 foto dari minimal 3 orang berbeda."
    )

    st.markdown("---")
    st.markdown("#### Format ZIP yang Diperlukan")
    st.code(
        "database.zip\n"
        "├── orang_1/\n"
        "│   ├── foto1.jpg  ← minimal 5 foto per orang\n"
        "│   └── foto2.jpg\n"
        "├── orang_2/\n"
        "│   └── ...\n"
        "└── orang_3/\n"
        "    └── ...",
        language=None
    )

    zip_file = st.file_uploader(
        "📦 Upload ZIP database wajah",
        type=["zip"],
        key="pub_zip"
    )

    if zip_file is not None:
        st.markdown("---")
        if st.button("🚀 Proses & Latih Model Sementara", type="primary",
                     use_container_width=True, key="btn_pub_train"):

            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract ZIP
                try:
                    with zipfile.ZipFile(zip_file, "r") as zf:
                        zf.extractall(tmpdir)
                except Exception as e:
                    st.error(f"ZIP tidak valid: {e}")
                    st.stop()

                tmp_path = Path(tmpdir)
                # Cari subfolder
                person_dirs = [d for d in tmp_path.rglob("*") if d.is_dir()]
                person_dirs = [d for d in person_dirs
                               if any(f.suffix.lower() in (".jpg",".jpeg",".png")
                                      for f in d.iterdir() if f.is_file())]

                if not person_dirs:
                    st.error("Tidak ada subfolder berisi foto. Periksa struktur ZIP.")
                    st.stop()

                # Ekstrak embedding
                all_embs, all_labels = [], []
                progress = st.progress(0, text="Mengekstrak embedding ArcFace...")
                total_files = sum(
                    len([f for f in d.iterdir()
                         if f.suffix.lower() in (".jpg",".jpeg",".png")])
                    for d in person_dirs
                )
                count = 0

                for pdir in sorted(person_dirs):
                    label = pdir.name
                    imgs = [f for f in sorted(pdir.iterdir())
                            if f.suffix.lower() in (".jpg",".jpeg",".png")]
                    for img_path in imgs:
                        img = cv2.imread(str(img_path))
                        if img is None: continue
                        emb, _ = get_embedding_safe(embedder, img)
                        if emb is not None:
                            all_embs.append(emb)
                            all_labels.append(label)
                        count += 1
                        progress.progress(
                            min(count / max(total_files, 1), 1.0),
                            text=f"Memproses {label}/{img_path.name}"
                        )
                progress.empty()

                if len(all_embs) < 4:
                    st.error("Terlalu sedikit foto valid (minimal 4 foto terdeteksi wajahnya).")
                    st.stop()

                # Fit PCA sementara
                from sklearn.decomposition import PCA as SklearnPCA
                X_pub = np.array(all_embs)
                n_comp = min(0.95, len(X_pub) - 1, X_pub.shape[1])
                pca_pub = SklearnPCA(n_components=min(len(X_pub)-1, 50),
                                     svd_solver="full")
                pca_pub.fit(X_pub)

                # Simpan ke session state
                st.session_state["pub_embs"]   = all_embs
                st.session_state["pub_labels"] = all_labels
                st.session_state["pub_pca"]    = pca_pub
                st.session_state["pub_ids"]    = sorted(set(all_labels))

                st.success(
                    f"✅ Selesai!\n\n"
                    f"- **{len(all_embs)} foto** berhasil diproses\n"
                    f"- **{len(set(all_labels))} identitas:** {', '.join(sorted(set(all_labels)))}\n"
                    f"- **{pca_pub.n_components_} komponen PCA** "
                    f"({pca_pub.explained_variance_ratio_.sum()*100:.1f}% variance)"
                )

    if "pub_pca" in st.session_state:
        st.markdown("---")
        st.markdown("#### Bandingkan Dua Foto dengan Model Sementara")

        from src.scorer import HybridScorer, ScoreResult, THRESHOLD_MATCH

        pub_c1, pub_c2 = st.columns(2)
        with pub_c1:
            pub_f1 = st.file_uploader("📁 Foto #1", type=["jpg","jpeg","png"], key="pub_f1")
            if pub_f1: st.image(pub_f1, use_container_width=True)
        with pub_c2:
            pub_f2 = st.file_uploader("📁 Foto #2", type=["jpg","jpeg","png"], key="pub_f2")
            if pub_f2: st.image(pub_f2, use_container_width=True)

        if pub_f1 and pub_f2:
            if st.button("🔍 Bandingkan (Model Sementara)", type="primary",
                         use_container_width=True, key="btn_pub_cmp"):
                with st.spinner("Mengekstrak embedding..."):
                    img1 = pil_to_bgr(Image.open(pub_f1))
                    img2 = pil_to_bgr(Image.open(pub_f2))
                    e1, c1 = get_embedding_safe(embedder, img1)
                    e2, c2 = get_embedding_safe(embedder, img2)

                if e1 is not None and e2 is not None:
                    pca_pub = st.session_state["pub_pca"]

                    # Hitung similarity
                    sim_full = float(np.dot(e1, e2))
                    p1 = pca_pub.transform([e1])[0]
                    p2 = pca_pub.transform([e2])[0]
                    p1n = p1 / (np.linalg.norm(p1) + 1e-10)
                    p2n = p2 / (np.linalg.norm(p2) + 1e-10)
                    sim_pca_pub = float(np.dot(p1n, p2n))
                    hybrid_pub  = WEIGHT_ONNX * sim_full + WEIGHT_PCA * sim_pca_pub
                    is_match_pub = hybrid_pub >= THRESHOLD_MATCH

                    from src.scorer import get_confidence, ScoreResult
                    conf, icon = get_confidence(hybrid_pub)
                    result_pub = ScoreResult(
                        sim_onnx=round(sim_full, 4),
                        sim_pca=round(sim_pca_pub, 4),
                        hybrid_score=round(hybrid_pub, 4),
                        eucl_dist=round(float(np.linalg.norm(p1-p2)), 4),
                        is_match=is_match_pub,
                        confidence=conf,
                        confidence_icon=icon,
                    )

                    # Buat scorer dummy untuk render (pakai pca sementara)
                    class _TmpScorer:
                        _pca = pca_pub
                    st.divider()
                    render_result(result_pub, e1, e2, c1, c2,
                                  _TmpScorer(), "Foto #1", "Foto #2")

        if st.button("🗑️ Reset Model Sementara", key="btn_pub_reset"):
            for k in ["pub_pca", "pub_embs", "pub_labels", "pub_ids"]:
                st.session_state.pop(k, None)
            st.rerun()
