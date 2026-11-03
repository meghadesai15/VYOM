"""
benchmark_rpi.py  ← copy this to RPi4 and run there
Benchmarks inference FPS on RPi4 using the camera or a test video.
Run on RPi4 ONLY.
Usage: python3 benchmark_rpi.py --source camera
       python3 benchmark_rpi.py --source test_clip.mp4
Run from the repo root (paths below are relative, same convention as the
rest of scripts/) — e.g. `python3 benchmark_rpi.py` from ~/vyom on the Pi.
"""

import time
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import deque

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from ai_edge_litert import interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

REPO_ROOT   = Path(__file__).resolve().parent.parent
MODEL_PATH  = str(REPO_ROOT / "models" / "vyom_int8.tflite")
LABELS_PATH = str(REPO_ROOT / "models" / "label_encoder_classes.npy")
N_FRAMES    = 45
IMG_SIZE    = 160   # matches the trained model input (was 224 — mismatched the model, would crash or silently misbehave)
N_THREADS   = 4   # RPi4 has 4 cores — use all

# FSM thresholds
CONF_SUSPICIOUS  = 0.65
CONF_AGGRESSIVE  = 0.80
CONSECUTIVE_NEED = 3   # frames in a row before state changes

# Updated for the 11-class retrain (Fast_Approach/Repeated_Looking/Running_Away
# now have training data — see RETRAIN_11CLASS_GUIDE.md).
SUSPICIOUS_CLASSES = {"Following_Person", "Fast_Approach", "Repeated_Looking"}
AGGRESSIVE_CLASSES = {"Aggressive_Gesture", "Forceful_Grab", "Physical_Push", "Running_Away"}

def load_model():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Generate it with `python scripts/convert_tflite.py` "
            f"(in an environment with TensorFlow installed) and copy it onto the Pi first."
        )
    interp = tflite.Interpreter(model_path=MODEL_PATH,
                                 num_threads=N_THREADS)
    interp.allocate_tensors()
    return interp

def preprocess_frame(frame):
    f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return f.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="camera",
                        help="'camera' or path to video file")
    args = parser.parse_args()

    print("Loading TFLite model...")
    interp = load_model()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    classes = np.load(LABELS_PATH, allow_pickle=True)

    cap = cv2.VideoCapture(0 if args.source == "camera" else args.source)
    if not cap.isOpened():
        print("ERROR: Cannot open video source")
        return

    # Sliding window buffer of frames
    frame_buffer = deque(maxlen=N_FRAMES)

    # FSM state tracking
    fsm_state       = "NORMAL"
    fsm_consecutive = 0
    inference_times = []
    fps_history     = []

    print(f"\nRunning benchmark (N_THREADS={N_THREADS})...")
    print("Press Ctrl+C to stop.\n")

    frame_count = 0
    t_start_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(preprocess_frame(frame))

            if len(frame_buffer) < N_FRAMES:
                continue   # fill buffer first

            # Run inference
            seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis, ...]
            t0 = time.perf_counter()
            interp.set_tensor(inp_idx, seq)
            interp.invoke()
            probs = interp.get_tensor(out_idx)[0]
            t1 = time.perf_counter()

            inf_ms = (t1 - t0) * 1000
            inference_times.append(inf_ms)

            pred_idx  = np.argmax(probs)
            pred_cls  = classes[pred_idx]
            conf      = float(probs[pred_idx])

            # FPS (sliding window)
            elapsed = time.time() - t_start_total
            fps     = frame_count / elapsed if elapsed > 0 else 0
            fps_history.append(fps)

            # FSM logic
            if pred_cls in AGGRESSIVE_CLASSES and conf >= CONF_AGGRESSIVE:
                candidate = "AGGRESSIVE"
            elif pred_cls in SUSPICIOUS_CLASSES and conf >= CONF_SUSPICIOUS:
                candidate = "SUSPICIOUS"
            else:
                candidate = "NORMAL"

            if candidate == fsm_state:
                fsm_consecutive += 1
            else:
                if fsm_consecutive >= CONSECUTIVE_NEED:
                    fsm_state = candidate
                    fsm_consecutive = 0
                    print(f"  ⚡ FSM state → {fsm_state}  (class={pred_cls}, conf={conf:.2f})")
                else:
                    fsm_consecutive += 1

            if frame_count % 30 == 0:
                avg_inf = np.mean(inference_times[-30:])
                print(f"Frame {frame_count:4d} | FPS: {fps:.1f} | "
                      f"Inference: {avg_inf:.0f}ms | "
                      f"Pred: {pred_cls:<22} ({conf:.2f}) | "
                      f"FSM: {fsm_state}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    # Summary
    print("\n=== Benchmark Summary ===")
    print(f"Total frames processed : {frame_count}")
    print(f"Average FPS            : {np.mean(fps_history):.1f}")
    print(f"Average inference time : {np.mean(inference_times):.1f} ms")
    print(f"Min/Max inference time : {min(inference_times):.1f} / {max(inference_times):.1f} ms")

    avg_fps = np.mean(fps_history)
    if avg_fps >= 10:
        print(f"\n✓ SRS PR-7 MET: {avg_fps:.1f} FPS ≥ 10 FPS target")
    else:
        print(f"\n⚠ SRS PR-7 NOT MET: {avg_fps:.1f} FPS < 10 FPS")
        print("  Try: reduce IMG_SIZE to 160, reduce N_FRAMES to 30")

if __name__ == "__main__":
    main()