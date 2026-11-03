"""
cut_clips.py  — VYOM Dataset Clip Cutter
Cuts ALL classes at once from D:/data/<class_folder>/ into 3-second clips.
Saves output to D:/data_clips/<class_name>/

Your exact folder names from screenshot are hardcoded below.
No loop needed — just run:  python cut_clips.py

Optional flags:
  --duration 3          clip length in seconds (default: 3)
  --max 80              max clips per class (default: 80)
  --data_dir D:/data    source root (default: D:/data)
  --out_dir  D:/data_clips  output root (default: D:/data_clips)
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

# ── Exact mapping: output class name → your actual folder name on disk ──────
# UPDATED for the 11-class retrain (see RETRAIN_11CLASS_GUIDE.md). The 8 old
# entries below still point at the old 8-class folder names — the 3 new ones
# are placeholders: fill in your real data/raw_v2 subfolder names once the
# new dataset is staged (see guide step 2/3, including the open "assault"
# mapping decision for Running_Away).
CLASS_MAP = {
    "Aggressive_Gesture":  "aggressive_gesture",
    "Forceful_Grab":       "forceful_grab",
    "Physical_Push":       "physical_push",
    "Following_Person":    "following_person",
    "Normal_Walking":      "normal_walking",
    "Standing_Normal":     "standing_normal",
    "Sitting_Normal":      "sitting_normal",
    "Talking_Normal":      "talking_normal",
    "Fast_Approach":       "<fill in your raw_v2 folder name>",
    "Repeated_Looking":    "<fill in your raw_v2 folder name>",
    "Running_Away":        "<fill in your raw_v2 folder name — or your 'assault' decision>",
}

# Per-class max clips. The old 70-80 caps were tuned for the old 80-clip-per-
# class target and will silently truncate the new, richer dataset. Raised to
# 999 (effectively uncapped) for all 11 classes — see the storage warning in
# RETRAIN_11CLASS_GUIDE.md (uncapped extraction can reach ~55-60GB of frames).
# Lower this per-class if your disk can't hold that.
PER_CLASS_MAX = {raw_folder: 999 for raw_folder in CLASS_MAP.values()}


def cut_single_video(video_path: Path, out_dir: Path,
                     clip_duration: int, max_clips: int) -> int:
    """
    Cuts one video into clips of `clip_duration` seconds.
    Returns number of clips written.
    Skips clips that already exist (safe to re-run).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    SKIP (cannot open): {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"    SKIP (bad FPS=0): {video_path.name}")
        cap.release()
        return 0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        print(f"    SKIP (zero dimensions): {video_path.name}")
        cap.release()
        return 0

    frames_per_clip = max(1, int(fps * clip_duration))
    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_clips       = total_frames // frames_per_clip

    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    written = 0

    for i in range(num_clips):
        if written >= max_clips:
            break

        out_path = out_dir / f"{video_path.stem}_c{i:04d}.mp4"
        if out_path.exists():
            written += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_per_clip)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frames_written = 0
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1

        writer.release()

        # Delete clips that are too short (less than half the target duration)
        if frames_written < frames_per_clip // 2:
            out_path.unlink(missing_ok=True)
        else:
            written += 1

    cap.release()
    return written


def process_all_classes(data_dir: Path, out_dir: Path,
                         clip_duration: int, global_max: int):
    """
    Iterates through all classes in CLASS_MAP and cuts clips for each.
    """
    print(f"\n{'='*60}")
    print(f"  VYOM Clip Cutter — processing {len(CLASS_MAP)} classes")
    print(f"  Source : {data_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Clip duration : {clip_duration}s | Max per class : {global_max}")
    print(f"{'='*60}\n")

    summary = []

    for cls_name, raw_folder in CLASS_MAP.items():
        in_dir      = data_dir / raw_folder
        cls_out_dir = out_dir  / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        # Use per-class max, but respect global override if provided
        max_clips = PER_CLASS_MAX.get(cls_name, 80)
        if global_max is not None:
            max_clips = min(max_clips, global_max)

        if not in_dir.exists():
            print(f"[MISSING] {cls_name:<30} ← folder not founD: {in_dir}")
            summary.append((cls_name, 0, 0, "MISSING"))
            continue

        # Collect all video files
        videos = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI"):
            videos.extend(in_dir.glob(ext))
        videos = list(set(videos))   # deduplicate

        if not videos:
            print(f"[EMPTY]   {cls_name:<30} ← no videos in {in_dir}")
            summary.append((cls_name, 0, 0, "EMPTY"))
            continue

        print(f"[{cls_name}]  {len(videos)} source videos → target {max_clips} clips")

        total_written = 0
        for vp in tqdm(videos, desc=f"  {cls_name}", unit="vid", leave=False):
            remaining = max_clips - total_written
            if remaining <= 0:
                break
            total_written += cut_single_video(vp, cls_out_dir,
                                              clip_duration, remaining)

        existing = len(list(cls_out_dir.glob("*.mp4")))
        status   = "OK" if existing >= 60 else ("LOW" if existing > 0 else "EMPTY")
        print(f"  → {existing} clips saved  [{status}]")
        summary.append((cls_name, len(videos), existing, status))

    # ── Final summary table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Class':<30} {'Raw vids':>8}  {'Clips out':>9}  Status")
    print(f"  {'-'*55}")

    total_clips = 0
    for cls_name, raw_count, clip_count, status in summary:
        flag = "  ← NEEDS MORE DATA" if status == "LOW"     else \
               "  ← FOLDER MISSING"  if status == "MISSING" else \
               "  ← NO VIDEOS"       if status == "EMPTY"   else ""
        print(f"  {cls_name:<30} {raw_count:>8}  {clip_count:>9}  {status}{flag}")
        total_clips += clip_count

    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<30} {'':>8}  {total_clips:>9}")
    print(f"\n  Clips saved to: {out_dir}")
    print(f"  Next step: run  python extract_frames.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="VYOM — Cut all dataset videos into fixed-length clips"
    )
    parser.add_argument(
        "--data_dir", type=str, default="D:/Vyom_AI/data/raw",
        help="Root folder containing your class subfolders (default: D:/Vyom_AI/data/raw)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="D:/Vyom_AI/data/clips",
        help="Output root folder for clips (default: D:/Vyom_AI/data/clips)"
    )
    parser.add_argument(
        "--duration", type=int, default=3,
        help="Clip duration in seconds (default: 3)"
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Override max clips per class (default: uses PER_CLASS_MAX table)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        print(f"\nERROR: data_dir not founD: {data_dir}")
        print(f"  Make sure your path is correct. Example:")
        print(f"  python cut_clips.py --data_dir D:/data --out_dir D:/data_clips")
        return

    process_all_classes(data_dir, out_dir, args.duration, args.max)


if __name__ == "__main__":
    main()