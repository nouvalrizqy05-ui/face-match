"""
src/embedder.py
===============
Ekstraktor embedding wajah menggunakan ArcFace via DeepFace.

ArcFace dipilih karena:
- Dilatih dengan AM-Softmax loss pada hipersfer (L2-normalized)
- Robust terhadap perubahan usia (paper: ArcFace on IJB-C aging benchmark)
- Output: vektor 512-dim yang merepresentasikan geometri wajah
  (jarak antar landmark: mata, hidung, mulut — invariant terhadap usia)

Referensi: Deng et al., "ArcFace: Additive Angular Margin Loss for
           Deep Face Recognition", CVPR 2019
"""

import os
import warnings
import numpy as np
import cv2
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")


class FaceEmbedder:
    """
    Mengekstrak ArcFace embedding dari gambar wajah.

    Usage:
        embedder = FaceEmbedder()
        emb, crop = embedder.get_embedding(image_bgr)
        # emb: np.ndarray shape (512,) — unit-normalized ArcFace vector
        # crop: np.ndarray — crop wajah yang dideteksi (untuk display)
    """

    def __init__(self, detector_backend: str = "opencv"):
        """
        Parameters
        ----------
        detector_backend : str
            Backend deteksi wajah untuk DeepFace.
            Pilihan: 'opencv', 'retinaface', 'mtcnn', 'ssd'
            Default 'opencv' karena tidak butuh download tambahan.
        """
        self.detector_backend = detector_backend
        self._deepface = None
        self._initialized = False

    def _init_deepface(self):
        """Lazy init — DeepFace hanya di-import saat pertama kali dipakai."""
        if self._initialized:
            return True
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._initialized = True
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def get_embedding(
        self,
        image_bgr: np.ndarray,
        enforce_detection: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Ekstrak ArcFace embedding dari satu frame gambar (BGR, format OpenCV).

        Returns
        -------
        (embedding, face_crop)
            embedding  : np.ndarray (512,) unit-normalized, atau None jika gagal
            face_crop  : np.ndarray gambar wajah yang dideteksi (RGB 112×112), atau None
        """
        if not self._init_deepface():
            raise RuntimeError(
                "DeepFace tidak tersedia.\n"
                "Install dengan: pip install deepface tf-keras"
            )

        try:
            # DeepFace.represent mengharapkan BGR atau path
            result = self._deepface.represent(
                img_path=image_bgr,
                model_name="ArcFace",
                enforce_detection=enforce_detection,
                detector_backend=self.detector_backend,
                align=True,                   # face alignment (eye-nose-mouth)
            )

            emb = np.array(result[0]["embedding"], dtype=np.float32)
            # ArcFace seharusnya sudah L2-normalized, tapi kita pastikan lagi
            emb = emb / (np.linalg.norm(emb) + 1e-10)

            # Crop wajah untuk ditampilkan
            face_crop = self._extract_crop(image_bgr, result[0])

            return emb, face_crop

        except Exception as e:
            error_msg = str(e).lower()
            if "face could not be detected" in error_msg or "no face" in error_msg:
                return None, None
            raise RuntimeError(f"ArcFace error: {e}")

    def get_embedding_from_path(self, image_path: str) -> tuple:
        """Versi path-based — lebih efisien untuk file di disk."""
        if not self._init_deepface():
            raise RuntimeError("DeepFace tidak tersedia.")
        try:
            result = self._deepface.represent(
                img_path=image_path,
                model_name="ArcFace",
                enforce_detection=True,
                detector_backend=self.detector_backend,
                align=True,
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)

            img = cv2.imread(image_path)
            face_crop = self._extract_crop(img, result[0]) if img is not None else None
            return emb, face_crop
        except Exception as e:
            if "face could not be detected" in str(e).lower():
                return None, None
            raise

    def _extract_crop(self, image_bgr: np.ndarray, result_dict: dict) -> np.ndarray | None:
        """Potong area wajah dari image sesuai facial_area dari DeepFace."""
        try:
            fa = result_dict.get("facial_area", {})
            x = fa.get("x", 0)
            y = fa.get("y", 0)
            w = fa.get("w", image_bgr.shape[1])
            h = fa.get("h", image_bgr.shape[0])

            # Padding sedikit
            pad = int(min(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image_bgr.shape[1], x + w + pad)
            y2 = min(image_bgr.shape[0], y + h + pad)

            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                return None
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (112, 112))
            return crop_resized
        except Exception:
            return None

    def is_available(self) -> bool:
        """Cek apakah DeepFace + ArcFace bisa digunakan."""
        return self._init_deepface()
