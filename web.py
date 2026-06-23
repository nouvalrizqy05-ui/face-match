import os
import io
import sys
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.scorer import HybridScorer
from src.embedder import FaceEmbedder
from src.visualizer import (
    plot_hybrid_score,
    plot_pca_variance,
    plot_embedding_comparison,
    plot_face_crops,
    plot_eigenface_overlay,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load models at startup
print("[FaceMatch] Loading models...")
try:
    scorer = HybridScorer("model/model_wajah.pkl")
    embedder = FaceEmbedder(detector_backend="opencv")
    # Pre-warm DeepFace (first call is slow)
    print("[FaceMatch] Pre-warming ArcFace model...")
    embedder._init_deepface()
    print("[FaceMatch] Models loaded successfully!")
except Exception as e:
    print(f"[FaceMatch] ERROR loading models: {e}")
    traceback.print_exc()
    scorer = None
    embedder = None


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", transparent=True)
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204  # No content, stop 404 error


@app.route("/compare", methods=["POST"])
def compare():
    if not scorer or not embedder:
        return jsonify({"error": "Sistem belum siap (Model gagal dimuat). Restart server."}), 500

    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "Dua file gambar wajib diunggah"}), 400

    file1 = request.files["file1"]
    file2 = request.files["file2"]

    if file1.filename == '' or file2.filename == '':
        return jsonify({"error": "Pilih dua gambar terlebih dahulu"}), 400

    try:
        print(f"[FaceMatch] Processing: {file1.filename} vs {file2.filename}")

        # Read files into cv2 format (BGR)
        file1_bytes = file1.read()
        file2_bytes = file2.read()

        print(f"[FaceMatch] File sizes: {len(file1_bytes)} bytes, {len(file2_bytes)} bytes")

        nparr1 = np.frombuffer(file1_bytes, np.uint8)
        nparr2 = np.frombuffer(file2_bytes, np.uint8)

        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            return jsonify({"error": "Format file tidak didukung. Gunakan JPG/PNG."}), 400

        # Resize large images to prevent OpenCV OOM crash
        def safe_resize(img, max_dim=1024):
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"[FaceMatch] Resized from {w}x{h} to {new_w}x{new_h}")
            return img

        img1 = safe_resize(img1)
        img2 = safe_resize(img2)

        print(f"[FaceMatch] Image shapes: {img1.shape}, {img2.shape}")

        # Extract embeddings
        print("[FaceMatch] Extracting ArcFace embedding 1...")
        emb1, crop1 = embedder.get_embedding(img1, enforce_detection=True)
        print("[FaceMatch] Extracting ArcFace embedding 2...")
        emb2, crop2 = embedder.get_embedding(img2, enforce_detection=True)

        if emb1 is None or emb2 is None:
            return jsonify({
                "error": "Wajah tidak terdeteksi pada salah satu atau kedua foto. Pastikan wajah terlihat jelas menghadap depan."
            }), 400

        print("[FaceMatch] Computing hybrid score...")
        result = scorer.score(emb1, emb2)
        print(f"[FaceMatch] Result: hybrid={result.hybrid_score}, match={result.is_match}")

        # Generate plots
        print("[FaceMatch] Generating visualizations...")
        fig_crops = plot_face_crops(crop1, crop2, result, "Foto A", "Foto B")
        fig_hybrid = plot_hybrid_score(result)
        fig_pca_var = plot_pca_variance(scorer._pca)
        fig_pca2d = plot_embedding_comparison(emb1, emb2, scorer._pca, "Foto A", "Foto B")
        fig_eigen = plot_eigenface_overlay(crop1, crop2, scorer._pca, emb1, emb2, result)

        response = {
            "is_match": result.is_match,
            "similarity": result.hybrid_score,
            "distance": result.eucl_dist,
            "threshold": 0.50,
            "confidence": result.confidence,
            "sim_onnx": result.sim_onnx,
            "sim_pca": result.sim_pca,
            "plot_crops": fig_to_base64(fig_crops),
            "plot_hybrid": fig_to_base64(fig_hybrid),
            "plot_pca_var": fig_to_base64(fig_pca_var),
            "plot_pca2d": fig_to_base64(fig_pca2d),
            "plot_eigenface": fig_to_base64(fig_eigen),
        }

        print("[FaceMatch] Success! Sending response.")
        return jsonify(response)

    except Exception as e:
        print(f"[FaceMatch] ERROR during compare: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[FaceMatch] Starting server on http://0.0.0.0:{port} ...")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=False)
