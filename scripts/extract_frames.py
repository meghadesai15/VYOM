"""
extract_frames.py
Extracts frames from every clip at exactly TARGET_FPS.
Saves per-clip .npy files containing shape (N_FRAMES, 224, 224, 3).
Usage: python scripts/extract_frames.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_FPS   = 15
CLIP_SECONDS = 3
N_FRAMES     = TARGET_FPS * CLIP_SECONDS   # 45 frames per clip
IMG_SIZE = 160

CLASSES = [
    "Normal_Walking", "Standing_Normal", "Sitting_Normal", "Talking_Normal",
    "Fast_Approach", "Following_Person", "Repeated_Looking",
    "Aggressive_Gesture", "Forceful_Grab", "Physical_Push", "Running_Away",
]

def extract_clip(video_path, n_frames=N_FRAMES, size=IMG_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    arr = np.array(frames, dtype=np.float32) / 255.0   # normalize to [0,1]
    return arr   # shape: (45, 224, 224, 3)

def main():
    skipped = 0
    total_saved = 0

    for cls in CLASSES:
        clip_dir  = Path("data/clips") / cls
        frame_dir = Path("data/frames") / cls
        frame_dir.mkdir(parents=True, exist_ok=True)

        clips = list(clip_dir.glob("*.mp4")) + list(clip_dir.glob("*.avi"))
        print(f"\n{cls}: {len(clips)} clips")

        for cp in tqdm(clips, desc=cls):
            out_path = frame_dir / (cp.stem + ".npy")
            if out_path.exists():
                total_saved += 1
                continue

            arr = extract_clip(cp)
            if arr is None or arr.shape[0] < N_FRAMES // 2:
                print(f"  SKIP (too short): {cp.name}")
                skipped += 1
                continue

            np.save(str(out_path), arr)
            total_saved += 1

    print(f"\n✓ Extraction complete. {total_saved} sequences saved. {skipped} skipped.")

if __name__ == "__main__":
    main()