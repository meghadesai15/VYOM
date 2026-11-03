# download_insightface_models.py — run once
import insightface
from insightface.app import FaceAnalysis
import os

os.makedirs("models", exist_ok=True)

# This auto-downloads buffalo_l pack (~300MB) into ~/.insightface/models/
# buffalo_l = ArcFace ResNet50 — best accuracy for outdoor/drone use
app = FaceAnalysis(
    name="buffalo_l",
    root="models",                    # saves to ./models/buffalo_l/ instead of home dir
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace buffalo_l ready.")
print("Models saved to: models/buffalo_l/")

# What was downloaded:
# models/buffalo_l/det_10g.onnx        — RetinaFace detector
# models/buffalo_l/w600k_r50.onnx      — ArcFace ResNet50 (512-D embeddings)
# models/buffalo_l/genderage.onnx      — optional gender/age (you won't use this)
# models/buffalo_l/2d106det.onnx       — 106-point landmark detector