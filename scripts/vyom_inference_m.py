# vyom_inference_m.py — per-track FSM, dev (ONNX) + Raspberry Pi (NCNN + INT8 TFLite) modes
import cv2
import numpy as np
import time
import argparse
import warnings
import os
import sys
import platform
from collections import deque, defaultdict
from ultralytics import YOLO
from insightface.app import FaceAnalysis
warnings.filterwarnings("ignore")

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="VYOM inference — video file or webcam")
parser.add_argument(
    "--source", type=str, default="0",
    help="Video file path, or camera index (default: 0 = default webcam)"
)
parser.add_argument(
    "--no-display", action="store_true",
    help="Disable the live preview window (headless mode)"
)
parser.add_argument(
    "--record", type=str, default=None,
    help="Save the annotated (boxes + labels) output video to this path, "
         "e.g. runs/session1.mp4 — useful for sharing a run for analysis later"
)
parser.add_argument(
    "--log-file", type=str, default=None,
    help="Also write the full console log to this text file "
         "(default: same name as --record with a .log extension)"
)
parser.add_argument(
    "--pi", dest="pi", action="store_true", default=None,
    help="Force Raspberry Pi mode (NCNN YOLO + INT8 TFLite classifier). "
         "Default: auto-detected from the device."
)
parser.add_argument(
    "--no-pi", dest="pi", action="store_false",
    help="Force desktop/dev mode (PyTorch YOLO + ONNX classifier), "
         "overriding auto-detection."
)
args = parser.parse_args()

# ── Platform detection ──────────────────────────────────────────────────────
def _is_raspberry_pi():
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "raspberry pi" in f.read().lower():
                return True
    except OSError:
        pass
    return platform.machine() in ("aarch64", "armv7l") and os.path.exists("/proc/device-tree")

ON_PI = _is_raspberry_pi() if args.pi is None else args.pi

# ── Optional console logging (tee stdout to a file) ────────────────────────
_log_path = args.log_file
if _log_path is None and args.record:
    _log_path = os.path.splitext(args.record)[0] + ".log"

_log_fh = None
if _log_path:
    os.makedirs(os.path.dirname(_log_path) or ".", exist_ok=True)
    _log_fh = open(_log_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = _Tee(sys.stdout, _log_fh)
    print(f"[VYOM] Logging console output to {_log_path}")

print(f"[VYOM] Platform: {'Raspberry Pi (NCNN YOLO + INT8 TFLite)' if ON_PI else 'Desktop (PyTorch YOLO + ONNX classifier)'}")

# ── Load models ───────────────────────────────────────────────────────────
YOLO_PT_PATH  = "models/yolov8n.pt"
YOLO_NCNN_DIR = "models/yolov8n_ncnn_model"

if ON_PI:
    if not os.path.isdir(YOLO_NCNN_DIR):
        print("[VYOM] NCNN export not found — attempting one-time export "
              "(needs the 'ncnn' package + internet on first run)...")
        try:
            YOLO(YOLO_PT_PATH).export(format="ncnn")
        except Exception as e:
            print(f"[VYOM] NCNN export failed ({e}); falling back to the "
                  f"PyTorch .pt model (slower, but will still run).")
    if os.path.isdir(YOLO_NCNN_DIR):
        print("[VYOM] Loading YOLOv8n (NCNN)...")
        yolo = YOLO(YOLO_NCNN_DIR)
    else:
        print("[VYOM] Loading YOLOv8n (PyTorch — NCNN unavailable)...")
        yolo = YOLO(YOLO_PT_PATH)
else:
    print("[VYOM] Loading YOLOv8n (PyTorch)...")
    yolo = YOLO(YOLO_PT_PATH)

# Always buffalo_l — buffalo_sc uses a different embedding space and would
# require re-enrolling the user, so we stay on buffalo_l on Pi too and just
# trim it down instead (see allowed_modules below).
FACE_MODEL = "buffalo_l"
DET_SIZE   = (192, 192) if ON_PI else (320, 320)
print(f"[VYOM] Loading InsightFace ({FACE_MODEL}, det_size={DET_SIZE})...")
face_app = FaceAnalysis(
    name=FACE_MODEL, root=".",
    # detection+recognition is all this pipeline uses; buffalo_l also ships
    # a 3D landmark model (144MB) and a gender/age model that we never call —
    # skipping them saves ~150MB of load and RAM with no change to embeddings
    # (they don't participate in the detect -> align -> embed path).
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

user_embedding = np.load("models/user_embedding.npy")
user_embedding = user_embedding / np.linalg.norm(user_embedding)

if ON_PI:
    from ai_edge_litert.interpreter import Interpreter
    TFLITE_PATH = "models/vyom_int8.tflite"
    if not os.path.isfile(TFLITE_PATH):
        raise FileNotFoundError(
            f"{TFLITE_PATH} not found. Generate it first with:\n"
            f"    python scripts/convert_tflite.py\n"
            f"(run that in an environment with TensorFlow installed, e.g. your dev venv, "
            f"then copy the resulting file onto the Pi)"
        )
    clf = Interpreter(model_path=TFLITE_PATH, num_threads=4)
    clf.allocate_tensors()
    inp_idx = clf.get_input_details()[0]["index"]
    out_idx = clf.get_output_details()[0]["index"]
    print("[VYOM] Classifier: INT8 TFLite (Pi mode)")
else:
    import onnxruntime as ort
    clf = ort.InferenceSession("models/vyom.onnx", providers=["CPUExecutionProvider"])
    input_name  = clf.get_inputs()[0].name
    output_name = clf.get_outputs()[0].name
    print("[VYOM] Classifier: ONNX Runtime (dev mode)")

classes = np.load("models/label_encoder_classes.npy", allow_pickle=True)
print(f"[VYOM] Classes ({len(classes)}): {list(classes)}")

# ── Config ────────────────────────────────────────────────────────────────
N_FRAMES         = 45
IMG_SIZE         = 160
FACE_CHECK_EVERY = 45 if ON_PI else 30   # widened on Pi — face check is expensive
CLASSIFY_EVERY   = 15 if ON_PI else 10   # widened on Pi — classify is expensive
YOLO_IMGSZ       = 256 if ON_PI else 320

FACE_THRESHOLD   = 0.55

CONF_SUSPICIOUS  = 0.65
CONF_AGGRESSIVE  = 0.80

# Rolling K-of-N vote (replaces the old strict-N-consecutive check): tolerates
# the classifier flickering back to NORMAL for a frame or two without erasing
# genuine sustained suspicious/aggressive signal. This matters more on Pi,
# where CLASSIFY_EVERY is wider, so "3 in a row" takes even longer to satisfy
# and any single flicker previously reset all progress.
FSM_WINDOW  = 5
FSM_ENTER_K = 3     # threat-tier hits (of last FSM_WINDOW) to enter/escalate
FSM_EXIT_K  = 4      # NORMAL hits (of last FSM_WINDOW) to exit a threat state — stricter than entry, on purpose

SUSPICIOUS_CLASSES = {"Following_Person", "Fast_Approach", "Repeated_Looking"}
AGGRESSIVE_CLASSES = {"Aggressive_Gesture", "Forceful_Grab", "Physical_Push", "Running_Away"}

# ── Per-track state ──────────────────────────────────────────────────────
class TrackState:
    def __init__(self, tid):
        self.tid          = tid
        self.fsm          = "NORMAL"      # NORMAL / SUSPICIOUS / AGGRESSIVE
        self.tier_history = deque(maxlen=FSM_WINDOW)
        self.frame_buf    = deque(maxlen=N_FRAMES)

    def update(self, pred_class, confidence):
        """
        Returns confirmed state change or None.
        Enter/escalate requires FSM_ENTER_K threat-tier hits in the last
        FSM_WINDOW predictions. Exit requires FSM_EXIT_K NORMAL hits in the
        last FSM_WINDOW predictions.
        """
        if pred_class in AGGRESSIVE_CLASSES and confidence >= CONF_AGGRESSIVE:
            tier = "AGGRESSIVE"
        elif pred_class in SUSPICIOUS_CLASSES and confidence >= CONF_SUSPICIOUS:
            tier = "SUSPICIOUS"
        else:
            tier = "NORMAL"

        self.tier_history.append(tier)
        hist       = self.tier_history
        agg_count  = sum(1 for t in hist if t == "AGGRESSIVE")
        susp_count = sum(1 for t in hist if t in ("AGGRESSIVE", "SUSPICIOUS"))
        norm_count = sum(1 for t in hist if t == "NORMAL")

        new_fsm = self.fsm
        if self.fsm == "NORMAL":
            if agg_count >= FSM_ENTER_K:
                new_fsm = "AGGRESSIVE"
            elif susp_count >= FSM_ENTER_K:
                new_fsm = "SUSPICIOUS"
        elif self.fsm == "SUSPICIOUS":
            if agg_count >= FSM_ENTER_K:
                new_fsm = "AGGRESSIVE"
            elif norm_count >= FSM_EXIT_K:
                new_fsm = "NORMAL"
        elif self.fsm == "AGGRESSIVE":
            if norm_count >= FSM_EXIT_K:
                new_fsm = "NORMAL"

        if new_fsm != self.fsm:
            self.fsm = new_fsm
            self.tier_history.clear()
            return new_fsm
        return None

# ── Track registry ────────────────────────────────────────────────────────
track_states  = {}    # tid → TrackState
user_track_id = None
frame_count   = 0

# ── Stats ─────────────────────────────────────────────────────────────────
class_counts     = defaultdict(int)
fsm_log          = []
total_frames     = 0

# ── Helpers ───────────────────────────────────────────────────────────────
def get_or_create_track(tid):
    if tid not in track_states:
        track_states[tid] = TrackState(tid)
    return track_states[tid]

def check_user_face(crop, tid):
    global user_track_id
    faces = face_app.get(crop)
    for face in faces:
        emb = face.embedding
        if emb is None:
            continue
        emb = emb / np.linalg.norm(emb)
        sim = np.dot(user_embedding, emb)
        if sim > FACE_THRESHOLD:
            if user_track_id != tid:
                print(f"[FACE] User locked → Track {tid} | sim={sim:.3f}")
            user_track_id = tid
            break

def run_classifier(frame_buffer):
    seq = np.array(frame_buffer, dtype=np.float32)[np.newaxis, ...]

    if ON_PI:
        clf.set_tensor(inp_idx, seq)
        clf.invoke()
        probs = clf.get_tensor(out_idx)[0]
    else:
        probs = clf.run([output_name], {input_name: seq})[0][0]

    return classes[np.argmax(probs)], float(np.max(probs))

def trigger_response(tid, state, pred_class, confidence, frame_no):
    ts = time.strftime("%H:%M:%S")
    if state == "SUSPICIOUS":
        print(f"[{ts}] ⚠  SUSPICIOUS | Track {tid} | "
              f"class={pred_class} conf={confidence:.2f} | alert sent")
    elif state == "AGGRESSIVE":
        print(f"[{ts}] 🚨 AGGRESSIVE | Track {tid} | "
              f"class={pred_class} conf={confidence:.2f} | ALARM + SOS")
    elif state == "NORMAL":
        print(f"[{ts}] ✓  NORMAL     | Track {tid} | threat cleared")
    fsm_log.append((frame_no, tid, state, pred_class, confidence))

# ── Main loop ─────────────────────────────────────────────────────────────
# --source accepts a plain integer (webcam index, e.g. "0") or a file/RTSP path.
SOURCE  = int(args.source) if args.source.isdigit() else args.source
IS_CAM  = isinstance(SOURCE, int)
DISPLAY = not args.no_display

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {SOURCE}")

ANNOTATE = DISPLAY or bool(args.record)

writer = None
if args.record:
    rec_fps = cap.get(cv2.CAP_PROP_FPS)
    if not rec_fps or rec_fps <= 1:
        rec_fps = 20.0
    rec_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    rec_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                              rec_fps, (rec_w, rec_h))
    print(f"[VYOM] Recording annotated output to {args.record}")

total_frames = 0 if IS_CAM else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[VYOM] Starting | model={FACE_MODEL} | source={'webcam ' + str(SOURCE) if IS_CAM else SOURCE}")
print(f"[VYOM] FSM: enter={FSM_ENTER_K}-of-{FSM_WINDOW} exit={FSM_EXIT_K}-of-{FSM_WINDOW} | "
      f"face_threshold={FACE_THRESHOLD}")
print(f"[VYOM] Intervals: classify_every={CLASSIFY_EVERY} face_check_every={FACE_CHECK_EVERY} yolo_imgsz={YOLO_IMGSZ}")
print(f"[VYOM] Thresholds: suspicious≥{CONF_SUSPICIOUS} aggressive≥{CONF_AGGRESSIVE}")
if IS_CAM and DISPLAY:
    print("[VYOM] Press 'q' in the preview window (or Ctrl+C) to stop.\n")
else:
    print()

results        = None
track_ids      = []
last_pred_text = defaultdict(str)
FSM_COLOR = {"NORMAL": (0, 200, 0), "SUSPICIOUS": (0, 200, 255), "AGGRESSIVE": (0, 0, 255)}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO every 2 frames
        if frame_count % 2 == 0 or results is None:
            results = yolo.track(
                source=frame,
                classes=[0],
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                imgsz=YOLO_IMGSZ
            )[0]

        if results.boxes.id is None:
            if DISPLAY:
                cv2.imshow("VYOM", frame)
            if writer is not None:
                writer.write(frame)
            if DISPLAY and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        track_ids  = results.boxes.id.int().cpu().tolist()
        boxes_xyxy = results.boxes.xyxy.cpu().tolist()
        confs      = results.boxes.conf.cpu().tolist()

        for tid, (x1, y1, x2, y2), det_conf in zip(track_ids, boxes_xyxy, confs):
            if det_conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue

            # Face check every N frames, OR immediately the first time we see
            # a track we've never seen before. Without this, a just-switched
            # track ID (including the enrolled user's, after a tracking loss)
            # runs through the "stranger" behavior classifier for up to
            # FACE_CHECK_EVERY frames before being (re)recognized.
            is_new_track = (tid not in track_states) and (tid != user_track_id)
            if is_new_track or frame_count % FACE_CHECK_EVERY == 0:
                check_user_face(crop, tid)

            is_user_track = (tid == user_track_id)

            if ANNOTATE:
                label = "USER" if is_user_track else last_pred_text.get(tid, "...")
                color = (255, 200, 0) if is_user_track else FSM_COLOR.get(
                    track_states[tid].fsm if tid in track_states else "NORMAL", (0, 200, 0)
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{tid} {label}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Skip the protected user
            if is_user_track:
                continue

            ts = get_or_create_track(tid)

            # Build frame buffer
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ts.frame_buf.append(resized)

            # Wait until buffer is full
            if len(ts.frame_buf) < N_FRAMES:
                continue

            # Only classify every CLASSIFY_EVERY frames
            if frame_count % CLASSIFY_EVERY != 0:
                continue

            t0 = time.perf_counter()

            pred_class, conf = run_classifier(ts.frame_buf)
            class_counts[pred_class] += 1
            last_pred_text[tid] = f"{pred_class} {conf:.2f}"

            new_state = ts.update(pred_class, conf)

            print(f"ACTION : {(time.perf_counter()-t0)*1000:.1f} ms")

            # Print per-track line with state prefix
            prefix = {"NORMAL": " ", "SUSPICIOUS": "⚠", "AGGRESSIVE": "🚨"}[ts.fsm]
            print(f" {prefix}[TRACK {tid:3d}] {pred_class:<25} conf={conf:.2f} | "
                f"FSM: {ts.fsm}")

            if new_state is not None:
                if new_state != "NORMAL":
                    print(f"  → FSM [{tid}]: {new_state} [{pred_class} @ {conf:.2f}]")
                trigger_response(tid, new_state, pred_class, conf, frame_count)

        if DISPLAY:
            cv2.imshow("VYOM", frame)
        if writer is not None:
            writer.write(frame)
        if DISPLAY and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clean dead tracks
        active = set(track_ids)
        for tid in [t for t in list(track_states) if t not in active]:
            del track_states[tid]

except KeyboardInterrupt:
    print("\n[VYOM] Stopped (Ctrl+C).")

cap.release()
if DISPLAY:
    cv2.destroyAllWindows()
if writer is not None:
    writer.release()
    print(f"[VYOM] Saved recording: {args.record}")

# ── Final verdict ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  VYOM — FINAL VERDICT")
print("="*65)
print(f"  Source     : {'webcam ' + str(SOURCE) if IS_CAM else SOURCE}")
print(f"  Frames     : {frame_count}")

if frame_count == 0:
    print("\n  No frames processed — exiting.")
    print("="*65)
    if _log_fh is not None:
        _log_fh.close()
    raise SystemExit(0)

total_classified = sum(class_counts.values())
tier_counts = {"Normal": 0, "Suspicious": 0, "Aggressive": 0}
print(f"\n  Class prediction counts:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    pct  = cnt / frame_count * 100
    bar  = "#" * int(pct * 0.8)
    tier = ("AGGR" if cls in AGGRESSIVE_CLASSES
            else "SUSP" if cls in SUSPICIOUS_CLASSES else "NORM")
    if tier == "AGGR": tier_counts["Aggressive"] += cnt
    elif tier == "SUSP": tier_counts["Suspicious"] += cnt
    else: tier_counts["Normal"] += cnt
    print(f"    [{tier}] {cls:<28} {cnt:4d} frames  ({pct:.1f}%)")

print(f"\n  Tier breakdown:")
for tier, cnt in tier_counts.items():
    pct = cnt / frame_count * 100
    print(f"    {tier:<12}: {cnt:4d} frames ({pct:.1f}%)")

dominant = max(class_counts, key=class_counts.get) if class_counts else "None"
dom_pct  = class_counts.get(dominant, 0) / frame_count * 100

agg_pct  = tier_counts["Aggressive"] / frame_count * 100
susp_pct = tier_counts["Suspicious"]  / frame_count * 100

print(f"\n  ── OVERALL VERDICT ──")
if agg_pct > 15:
    print(f"  🚨 HIGH THREAT — aggressive behavior dominated")
elif agg_pct > 5 or susp_pct > 20:
    print(f"  ⚡ MODERATE THREAT — suspicious/aggressive behavior detected")
elif susp_pct > 5:
    print(f"  ⚠  LOW THREAT — some suspicious behavior observed")
else:
    print(f"  ✓  SAFE — no significant threat detected")

print(f"  Dominant class : {dominant} ({dom_pct:.1f}%)")

print(f"\n  FSM transition log ({len(fsm_log)} events):")
for frame_no, tid, state, cls, conf in fsm_log:
    icon = {"NORMAL":"✓","SUSPICIOUS":"⚠","AGGRESSIVE":"🚨"}.get(state,"?")
    print(f"    Frame {frame_no:4d} | Track {tid:3d} | {icon} {state:<12} [{cls} @ {conf:.2f}]")
print("="*65)

if _log_fh is not None:
    _log_fh.close()
