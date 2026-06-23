"""
src/scorer.py
=============
Hybrid scoring engine untuk deteksi kemiripan wajah.

Formula hybrid (sesuai arsitektur BioID):
    sim_onnx = cosine_similarity(emb1_512, emb2_512)   ← ArcFace langsung
    sim_pca  = cosine_similarity(pca(emb1), pca(emb2)) ← setelah reduksi PCA
    hybrid   = sim_onnx × 0.70 + sim_pca × 0.30

Mengapa hybrid?
  - sim_onnx menangkap identitas wajah di ruang 512-dim penuh (presisi tinggi)
  - sim_pca menangkap pola variasi utama lintas usia dari training FGNET (robustness)
  - Kombinasi 70/30 meminimalkan false positive sambil mempertahankan recall

Threshold keputusan:
  - is_match : hybrid_score ≥ 0.50
  - Confidence levels: ≥0.70 Sangat Tinggi | 0.60-0.70 Tinggi |
                       0.50-0.60 Sedang | <0.50 Rendah
"""

import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------------
# KONFIGURASI THRESHOLD
# ------------------------------------------------------------------
THRESHOLD_MATCH      = 0.50   # minimum hybrid_score untuk dinyatakan MIRIP
WEIGHT_ONNX          = 0.70   # bobot cosine di ruang ArcFace 512-dim
WEIGHT_PCA           = 0.30   # bobot cosine di ruang PCA 95-dim

CONFIDENCE_LEVELS = [
    (0.70, "Sangat Tinggi", "🟢"),
    (0.60, "Tinggi",        "🟡"),
    (0.50, "Sedang",        "🟠"),
    (0.00, "Rendah",        "🔴"),
]


# ------------------------------------------------------------------
# DATACLASS HASIL
# ------------------------------------------------------------------
@dataclass
class ScoreResult:
    sim_onnx    : float    # cosine similarity di ruang 512-dim
    sim_pca     : float    # cosine similarity di ruang 95-dim
    hybrid_score: float    # skor akhir gabungan
    eucl_dist   : float    # jarak euclidean di ruang PCA (info tambahan)
    is_match    : bool     # keputusan akhir
    confidence  : str      # "Sangat Tinggi" / "Tinggi" / "Sedang" / "Rendah"
    confidence_icon: str   # emoji


def get_confidence(score: float) -> tuple[str, str]:
    for threshold, label, icon in CONFIDENCE_LEVELS:
        if score >= threshold:
            return label, icon
    return "Rendah", "🔴"


# ------------------------------------------------------------------
# HYBRID SCORER
# ------------------------------------------------------------------
class HybridScorer:
    """
    Menggabungkan ArcFace embedding langsung dan representasi PCA
    untuk scoring kemiripan wajah yang robust terhadap perubahan usia.

    Parameters
    ----------
    model_path : str
        Path ke model_wajah.pkl yang berisi sklearn PCA terlatih dari FGNET.
    """

    def __init__(self, model_path: str = "model/model_wajah.pkl"):
        self._pca = None
        self._model_meta = {}
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        mp = Path(self._model_path)
        if not mp.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {self._model_path}\n"
                "Jalankan train.py dulu, atau pastikan model_wajah.pkl ada di folder model/"
            )
        with open(mp, "rb") as f:
            data = pickle.load(f)
        self._pca = data["pca_model"]
        self._model_meta = data.get("metadata", {})

    def score(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> ScoreResult:
        """
        Hitung hybrid score antara dua ArcFace embedding 512-dim.

        Parameters
        ----------
        emb1, emb2 : np.ndarray shape (512,), unit-normalized

        Returns
        -------
        ScoreResult dengan semua metrik
        """
        # Pastikan unit-normalized
        e1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
        e2 = emb2 / (np.linalg.norm(emb2) + 1e-10)

        # 1. Cosine similarity di ruang 512-dim (ArcFace langsung)
        sim_onnx = float(np.dot(e1, e2))

        # 2. Transform ke ruang PCA 95-dim
        p1 = self._pca.transform([e1])[0]
        p2 = self._pca.transform([e2])[0]

        # Cosine similarity di ruang PCA
        p1_norm = p1 / (np.linalg.norm(p1) + 1e-10)
        p2_norm = p2 / (np.linalg.norm(p2) + 1e-10)
        sim_pca = float(np.dot(p1_norm, p2_norm))

        # 3. Euclidean distance di ruang PCA (info tambahan)
        eucl_dist = float(np.linalg.norm(p1 - p2))

        # 4. Hybrid score
        hybrid = WEIGHT_ONNX * sim_onnx + WEIGHT_PCA * sim_pca

        # 5. Keputusan
        is_match = hybrid >= THRESHOLD_MATCH
        confidence, icon = get_confidence(hybrid)

        return ScoreResult(
            sim_onnx=round(sim_onnx, 4),
            sim_pca=round(sim_pca, 4),
            hybrid_score=round(hybrid, 4),
            eucl_dist=round(eucl_dist, 4),
            is_match=is_match,
            confidence=confidence,
            confidence_icon=icon,
        )

    @property
    def pca_info(self) -> dict:
        """Info model PCA yang sedang dipakai."""
        if self._pca is None:
            return {}
        return {
            "n_components"     : self._pca.n_components_,
            "input_dim"        : self._pca.components_.shape[1],
            "variance_explained": float(self._pca.explained_variance_ratio_.sum()),
            "pc1_variance"     : float(self._pca.explained_variance_ratio_[0]),
            **self._model_meta,
        }
