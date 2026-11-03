import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    root=".",
    # matches vyom_inference_m.py — skips the unused 3D-landmark (144MB) and
    # gender/age models; doesn't change the resulting embedding.
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"]   # safer for most PCs
)

app.prepare(ctx_id=-1, det_size=(640,640))

embeddings = []

image_dir = "runs/test/user_face"

for file in os.listdir(image_dir):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, file)

    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Could not read {file}")
        continue

    faces = app.get(frame)

    if len(faces) != 1:
        print(f"Skipped {file} (faces found: {len(faces)})")
        continue

    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)

    embeddings.append(emb)

    print(f"Processed {file}")

if len(embeddings) == 0:
    raise ValueError("No valid face images found")

user_embedding = np.mean(embeddings, axis=0)
user_embedding /= np.linalg.norm(user_embedding)

np.save("models/user_embedding.npy", user_embedding)

print(f"\nEnrolled using {len(embeddings)} images")
print("Saved: models/user_embedding.npy")