# enroll_user.py — live webcam enrollment for the protected user
#
# Captures ArcFace embeddings via the webcam and saves the averaged,
# L2-normalized 512-D embedding to models/user_embedding.npy. This is the
# reference vector vyom_inference_m.py compares every tracked face against
# to lock onto (and skip classifying) the enrolled user.
#
# Usage: python scripts/enroll_user.py [--camera 0] [--samples 10]
import argparse
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# root="." → looks under ./models/buffalo_l, matching vyom_inference_m.py
# and where download_insightface_models.py actually placed the pack.
# (root="models" was a bug — it made insightface look under
#  models/models/buffalo_l and silently re-download the ~300MB pack there.)
# allowed_modules matches vyom_inference_m.py — skips the unused 3D-landmark
# (144MB) and gender/age models. Doesn't change the embedding: detection's
# own 5-point keypoints handle alignment, recognition produces the 512-D
# vector, neither depends on the modules being skipped.
app = FaceAnalysis(name="buffalo_l", root=".",
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Matches FACE_THRESHOLD in vyom_inference_m.py — keep these in sync.
FACE_THRESHOLD = 0.55


def enroll_user(frame_list):
    """
    frame_list: list of frames, ideally each containing exactly one face
                (the caller should already have filtered for this).
    Returns: 512-D embedding (numpy array) — saved as user_embedding.npy
    """
    embeddings = []
    for frame in frame_list:
        faces = app.get(frame)
        if len(faces) == 1:                        # exactly one face in frame
            embeddings.append(faces[0].embedding)  # 512-D ArcFace vector

    if not embeddings:
        raise ValueError("No face detected during enrollment")

    # Average across frames for a robust embedding
    user_embedding = np.mean(embeddings, axis=0)
    user_embedding /= np.linalg.norm(user_embedding)  # L2 normalize

    np.save("models/user_embedding.npy", user_embedding)
    print(f"\n[ENROLL] Done — averaged {len(embeddings)} samples.")
    print(f"[ENROLL] Saved to models/user_embedding.npy")
    return user_embedding


def is_user(frame, user_embedding, threshold=FACE_THRESHOLD):
    """
    Returns (True/False, similarity) for the closest face in frame vs.
    the enrolled user. threshold=0.55 matches vyom_inference_m.py.
    """
    faces = app.get(frame)
    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        similarity = np.dot(user_embedding, emb)   # cosine similarity
        if similarity > threshold:
            return True, similarity
    return False, 0.0


def run_enrollment(camera_index=0, n_samples=10):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print(f"[ENROLL] Camera {camera_index} open. Look straight at the camera.")
    print("[ENROLL] Capturing samples — hold still. Press 'q' to abort.\n")

    good_frames = []
    last_capture_t = 0
    import time
    while len(good_frames) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        # Only attempt a capture a few times a second so samples aren't
        # near-duplicates of each other.
        now = time.time()
        status = f"Samples: {len(good_frames)}/{n_samples}"
        faces = None
        if now - last_capture_t > 0.25:
            faces = app.get(frame)
            last_capture_t = now
            if faces is not None and len(faces) == 1:
                good_frames.append(frame.copy())
                status = f"Samples: {len(good_frames)}/{n_samples}  (captured)"
            elif faces is not None and len(faces) > 1:
                status = f"Samples: {len(good_frames)}/{n_samples}  (multiple faces — only one person should be in frame)"
            elif faces is not None:
                status = f"Samples: {len(good_frames)}/{n_samples}  (no face detected)"

        disp = frame.copy()
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("VYOM Enrollment — press q to abort", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[ENROLL] Aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    return enroll_user(good_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll the protected user via webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--samples", type=int, default=10, help="Number of face samples to average (default: 10)")
    args = parser.parse_args()

    run_enrollment(camera_index=args.camera, n_samples=args.samples)
