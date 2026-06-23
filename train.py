"""
train.py
========
Script pelatihan OFFLINE — jalankan sekali di PC kamu, hasilkan model_wajah.pkl

Pipeline:
  1. Baca semua foto dari FGNET/images/ (naming: 001A02.JPG → subjek 001)
  2. Ekstrak embedding 512-dim per foto via ArcFace (DeepFace)
  3. Kumpulkan jadi matriks X (n_samples × 512)
  4. Fit sklearn PCA (n_components=0.95) → ~95 komponen
  5. Simpan model/model_wajah.pkl

Jalankan:
    python train.py
    python train.py --fgnet-dir FGNET/images --output model/model_wajah.pkl
    python train.py --skip-failed   # lewati foto yang gagal deteksi

Catatan:
  - model_wajah.pkl dari BioID (repo referensi) SUDAH BISA DIPAKAI langsung
    sehingga langkah ini OPSIONAL jika kamu sudah punya model_wajah.pkl
  - Training ~955 foto memakan waktu 10-30 menit tergantung CPU
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

# Suppress TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Train PCA dari ArcFace embeddings (FGNET)")
    p.add_argument("--fgnet-dir",  default="FGNET/images",
                   help="Folder berisi foto FGNET dengan naming 001A02.JPG")
    p.add_argument("--output",     default="model/model_wajah.pkl")
    p.add_argument("--n-components", type=float, default=0.95)
    p.add_argument("--skip-failed", action="store_true",
                   help="Lewati foto yang gagal deteksi wajah")
    p.add_argument("--max-photos", type=int, default=None,
                   help="Batasi jumlah foto (untuk testing cepat)")
    return p.parse_args()


def extract_arcface_embedding(image_path: str, enforce_detection: bool = True):
    """
    Ekstrak ArcFace embedding 512-dim dari satu foto.
    Requires: pip install deepface tf-keras
    """
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=image_path,
        model_name="ArcFace",
        enforce_detection=enforce_detection,
        detector_backend="opencv",
    )
    emb = np.array(result[0]["embedding"], dtype=np.float32)
    # Normalize ke unit sphere (seperti ArcFace seharusnya)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


def collect_fgnet_embeddings(fgnet_dir: str, skip_failed: bool, max_photos=None):
    """
    Baca semua foto FGNET, ekstrak embedding ArcFace.
    Naming convention: 001A02.JPG → subject_id = '001'
    """
    fgnet_path = Path(fgnet_dir)
    if not fgnet_path.exists():
        print(f"[ERROR] FGNET dir tidak ditemukan: {fgnet_dir}")
        print("  Download dari: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("  Atau gunakan model_wajah.pkl yang sudah ada (skip training)")
        sys.exit(1)

    image_files = sorted([
        f for f in fgnet_path.iterdir()
        if f.suffix.upper() in (".JPG", ".JPEG", ".PNG", ".BMP")
    ])

    if max_photos:
        image_files = image_files[:max_photos]

    print(f"[INFO] Ditemukan {len(image_files)} foto di {fgnet_dir}")
    print("[INFO] Mengekstrak ArcFace embeddings (ini memakan waktu)...\n")

    embeddings, labels = [], []
    failed, success = 0, 0

    for i, f in enumerate(image_files):
        # Parse subject ID dari 3 karakter pertama filename
        subject_id = f.name[:3]

        try:
            emb = extract_arcface_embedding(str(f), enforce_detection=True)
            embeddings.append(emb)
            labels.append(subject_id)
            success += 1

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1:4d}/{len(image_files)}] ✓ {f.name} | "
                      f"subject={subject_id} | dim={len(emb)}")
        except Exception as e:
            failed += 1
            if not skip_failed:
                print(f"  [{i+1:4d}/{len(image_files)}] ✗ {f.name} | Error: {e}")
            if failed > 50 and not skip_failed:
                print("[WARN] Banyak foto gagal. Coba --skip-failed")

    print(f"\n[INFO] Selesai: {success} berhasil, {failed} gagal")
    return np.array(embeddings, dtype=np.float32), labels


def train_pca(X: np.ndarray, n_components=0.95):
    """
    Fit sklearn PCA dari matriks embedding.
    Input : X (n_samples × 512)
    Output: pca model dengan n_components_ komponen utama
    """
    from sklearn.decomposition import PCA

    print(f"\n[INFO] Fitting PCA...")
    print(f"  Input shape  : {X.shape}")
    print(f"  n_components : {n_components} (variance target)")

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    pca.fit(X)

    print(f"  Komponen     : {pca.n_components_}")
    print(f"  Variance     : {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  PC1 variance : {pca.explained_variance_ratio_[0]*100:.2f}%")
    return pca


def main():
    args = parse_args()

    print("=" * 60)
    print("  FACE COMPARISON — TRAINING PIPELINE")
    print("  ArcFace (512-dim) → PCA (95-dim)")
    print("=" * 60)
    print()

    # Cek apakah model sudah ada
    model_path = Path(args.output)
    if model_path.exists():
        print(f"[INFO] Model sudah ada: {args.output}")
        print("  Gunakan --output path_baru.pkl untuk retrain ke file berbeda")
        resp = input("  Lanjut retrain? (y/N): ").strip().lower()
        if resp != "y":
            print("  Training dibatalkan. Model yang ada tetap dipakai.")
            return

    # Langkah 1-2: Kumpulkan embedding FGNET
    X, labels = collect_fgnet_embeddings(
        args.fgnet_dir,
        skip_failed=args.skip_failed,
        max_photos=args.max_photos,
    )

    if len(X) < 10:
        print("[ERROR] Terlalu sedikit sampel valid. Periksa folder FGNET.")
        sys.exit(1)

    # Langkah 3: Fit PCA
    pca = train_pca(X, n_components=args.n_components)

    # Langkah 4: Simpan model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "pca_model": pca,
        "metadata": {
            "n_train_samples" : len(X),
            "n_labels"        : len(set(labels)),
            "input_dim"       : X.shape[1],
            "n_components"    : pca.n_components_,
            "variance"        : float(pca.explained_variance_ratio_.sum()),
            "embedding_model" : "ArcFace",
        },
    }
    with open(args.output, "wb") as f:
        pickle.dump(model_data, f)

    size_kb = model_path.stat().st_size / 1024
    print(f"\n[INFO] Model disimpan: {args.output} ({size_kb:.0f} KB)")
    print()
    print("=" * 60)
    print("  TRAINING SELESAI")
    print("=" * 60)
    print(f"  Sampel training  : {len(X)}")
    print(f"  Subjek FGNET     : {len(set(labels))}")
    print(f"  Input dim        : {X.shape[1]} (ArcFace)")
    print(f"  Komponen PCA     : {pca.n_components_}")
    print(f"  Variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  Model path       : {args.output}")
    print()
    print("Jalankan app: streamlit run app.py")


if __name__ == "__main__":
    main()
