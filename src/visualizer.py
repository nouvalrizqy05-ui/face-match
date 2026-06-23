"""
src/visualizer.py
=================
Visualisasi PCA untuk keperluan presentasi dan pemahaman dosen.

Fungsi-fungsi di sini menghasilkan matplotlib Figure yang bisa
langsung di-render di Streamlit via st.pyplot().
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_hybrid_score(result) -> plt.Figure:
    """
    Visualisasi dekomposisi hybrid score:
    Tampilkan sim_onnx, sim_pca, dan hybrid_score dalam bar chart.
    """
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_alpha(0)

    labels   = ["ArcFace\n(512-dim)", "PCA\n(95-dim)", "Hybrid\nScore"]
    values   = [result.sim_onnx, result.sim_pca, result.hybrid_score]
    weights  = [f"×{0.70:.0%}", f"×{0.30:.0%}", "Final"]
    colors   = ["#3498db", "#9b59b6", "#2ecc71" if result.is_match else "#e74c3c"]

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="white")

    # Threshold line
    ax.axvline(0.50, color="#f39c12", linestyle="--", linewidth=1.5,
               label="Threshold = 0.50")

    for bar, val, w in zip(bars, values, weights):
        ax.text(
            max(val - 0.04, 0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}  {w}",
            va="center", ha="right" if val > 0.08 else "left",
            fontsize=9.5, color="white", fontweight="bold"
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Similarity Score", fontsize=10)
    ax.set_title("Dekomposisi Hybrid Score", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_pca_variance(pca_model) -> plt.Figure:
    """
    Grafik explained variance ratio kumulatif.
    Menunjukkan berapa komponen PCA yang diperlukan.
    """
    ratios = pca_model.explained_variance_ratio_
    cumsum = np.cumsum(ratios)
    k = len(ratios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_alpha(0)

    # Left: cumulative
    ax1.plot(range(1, k + 1), cumsum * 100, color="#3498db", linewidth=2)
    ax1.axhline(95, color="#e74c3c", linestyle="--", linewidth=1.2,
                label="95% variance threshold")
    ax1.fill_between(range(1, k + 1), cumsum * 100, alpha=0.15, color="#3498db")
    ax1.set_xlabel("Jumlah Komponen PCA (k)")
    ax1.set_ylabel("Cumulative Variance (%)")
    ax1.set_title("Variance Kumulatif", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 105)

    # Right: individual (top 20)
    n_show = min(20, k)
    ax2.bar(range(1, n_show + 1), ratios[:n_show] * 100,
            color="#9b59b6", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Komponen PCA ke-")
    ax2.set_ylabel("Variance (%)")
    ax2.set_title(f"Variance per Komponen (Top {n_show})", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_embedding_comparison(emb1: np.ndarray, emb2: np.ndarray,
                               pca_model, label1="Foto 1", label2="Foto 2") -> plt.Figure:
    """
    Visualisasi dua embedding di ruang PCA 2D (PC1 vs PC2).
    Menunjukkan secara visual apakah dua wajah berada di area yang sama.
    """
    p1 = pca_model.transform([emb1])[0]
    p2 = pca_model.transform([emb2])[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_alpha(0)

    ax.scatter(p1[0], p1[1], c="#3498db", s=200, zorder=5, label=label1,
               edgecolors="white", linewidth=1.5)
    ax.scatter(p2[0], p2[1], c="#e74c3c", s=200, zorder=5, label=label2,
               edgecolors="white", linewidth=1.5)

    # Garis penghubung
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", alpha=0.4, linewidth=1.2)

    # Anotasi jarak
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    dist  = np.linalg.norm(p1[:2] - p2[:2])
    ax.annotate(f"dist={dist:.3f}", (mid_x, mid_y), fontsize=8.5,
                ha="center", va="bottom", color="#555")

    ax.set_xlabel("PC1 (komponen utama 1)", fontsize=9)
    ax.set_ylabel("PC2 (komponen utama 2)", fontsize=9)
    ax.set_title("Posisi Wajah di Ruang PCA 2D", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def plot_face_crops(crop1: np.ndarray, crop2: np.ndarray,
                    result, label1="Foto 1", label2="Foto 2") -> plt.Figure:
    """
    Tampilkan dua crop wajah berdampingan dengan skor di judul.
    """
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    fig.patch.set_alpha(0)

    color = "#2ecc71" if result.is_match else "#e74c3c"
    status = "✅ MIRIP" if result.is_match else "❌ BERBEDA"

    for ax, crop, title in zip(axes, [crop1, crop2], [label1, label2]):
        if crop is not None:
            ax.imshow(crop)
        else:
            ax.imshow(np.zeros((112, 112, 3), dtype=np.uint8))
            ax.text(56, 56, "Wajah\ntidak\nterdeteksi",
                    ha="center", va="center", color="white", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    fig.suptitle(
        f"{status}  |  Hybrid Score: {result.hybrid_score:.4f}  |  "
        f"{result.confidence_icon} {result.confidence}",
        fontsize=10, fontweight="bold", color=color
    )
    plt.tight_layout()
    return fig


def plot_eigenface_overlay(crop1: np.ndarray, crop2: np.ndarray,
                           pca_model, emb1: np.ndarray, emb2: np.ndarray,
                           result) -> plt.Figure:
    """
    Visualisasi gaya 'eigenface': menampilkan kedua wajah dalam grayscale,
    overlay/blend keduanya sebagai bayangan matriks, dan heatmap perbedaan.
    """
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.8),
                             gridspec_kw={'width_ratios': [1, 1, 1.2, 1], 'wspace': 0.12})
    fig.patch.set_alpha(0)

    def to_gray_112(crop):
        if crop is None:
            return np.zeros((112, 112), dtype=np.uint8)
        if len(crop.shape) == 3:
            gray = np.mean(crop, axis=2).astype(np.uint8)
        else:
            gray = crop.astype(np.uint8)
        if gray.shape[:2] != (112, 112):
            gray = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
        return gray

    g1 = to_gray_112(crop1)
    g2 = to_gray_112(crop2)

    color = "#2ecc71" if result.is_match else "#e74c3c"

    # Panel 1: Grayscale Foto A
    axes[0].imshow(g1, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Foto A", fontsize=8, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Grayscale Foto B
    axes[1].imshow(g2, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Foto B", fontsize=8, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Eigenface-style ghosted overlay (alpha blend)
    blended = (g1.astype(np.float32) * 0.5 + g2.astype(np.float32) * 0.5).astype(np.uint8)
    axes[2].imshow(blended, cmap='gray', vmin=0, vmax=255)
    sim = result.hybrid_score
    axes[2].set_title(f"Overlay ({sim:.1%})", fontsize=8, fontweight="bold", color=color)
    axes[2].axis("off")

    # Panel 4: Difference heatmap
    diff = np.abs(g1.astype(np.float32) - g2.astype(np.float32))
    diff = diff / (diff.max() + 1e-8) * 255
    axes[3].imshow(diff.astype(np.uint8), cmap='hot')
    axes[3].set_title("Peta Perbedaan", fontsize=8, fontweight="bold")
    axes[3].axis("off")

    status = "MIRIP" if result.is_match else "BERBEDA"
    fig.suptitle(f"Eigenface Analysis — {status}", fontsize=10, fontweight="bold", color=color)
    fig.subplots_adjust(top=0.82, bottom=0.02, left=0.02, right=0.98)
    return fig
