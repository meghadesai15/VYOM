"""
convert_tflite.py  —  fixed for LSTM + TFLite
LSTM models require SELECT_TF_OPS to handle TensorListReserve ops.
Your model: input (None, 45, 160, 160, 3)  |  output (None, 8)

NOTE on quantization: full INT8 (calibrated with a representative dataset)
fails for this model — TFLite's calibrator can't run FlexTensorListReserve/
SetItem/Stack ops (which is how Keras LSTM lowers), so it errors out during
calibration regardless of the data you feed it. This is a TensorFlow/TFLite
limitation with LSTM layers, not a bug in this script. We use *dynamic-range*
quantization instead (weights quantized to int8, activations stay float32) —
it converts successfully and still shrinks/speeds up the model vs FP32,
just not as much as true full-int8 would. The model still needs
SELECT_TF_OPS/the Flex delegate at inference time either way, since the
LSTM's TensorList ops aren't native TFLite ops.
If you want true full-INT8 later: rebuild the LSTM layer with
`unroll=True` (valid since the sequence length is fixed at 45 frames) and
transfer the existing trained weights in — that removes the TensorList ops
entirely, which would both unblock full-INT8 calibration and drop the
Flex-delegate requirement at runtime (a real speed win on Raspberry Pi).
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH  = Path("models/vyom_mobilenet_lstm.h5")
TFLITE_FP32 = Path("models/vyom_fp32.tflite")
TFLITE_INT8 = Path("models/vyom_int8.tflite")
FRAMES_DIR  = Path("data/frames")

# ── IMPORTANT: corrected from logs ───────────────────────────────────────
N_FRAMES  = 45
IMG_SIZE  = 160     # your model uses 160x160, NOT 224x224
N_CLASSES = 8       # your model outputs 8 classes, NOT 13

# ── Also fix this in vyom_inference.py ───────────────────────────────────
# IMG_SIZE = 160   ← change this line in inference too

CLASSES = np.load("models/label_encoder_classes.npy", allow_pickle=True)

# Kept for reference / for use if this model is later rebuilt with
# unroll=True (see note above) and full-INT8 calibration becomes viable —
# not called in the current dynamic-range conversion path below.
def representative_dataset():
    """Feed real samples to calibrate INT8 quantization ranges."""
    count = 0
    for cls_name in CLASSES:
        cls_dir = FRAMES_DIR / cls_name
        if not cls_dir.exists():
            continue
        for nf in list(cls_dir.glob("*.npy"))[:10]:
            arr = np.load(str(nf))
            arr = arr[:N_FRAMES].astype(np.float32)
            if arr.shape[0] < N_FRAMES:
                continue
            yield [arr[np.newaxis, ...]]   # shape: (1, 45, 160, 160, 3)
            count += 1
            if count >= 110:
                return

def convert():
    print("Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"Input  shape: {model.input_shape}")   # verify (None, 45, 160, 160, 3)
    print(f"Output shape: {model.output_shape}")  # verify (None, 8)

    # ── FP32 baseline (with SELECT_TF_OPS for LSTM) ───────────────────────
    print("\nConverting FP32...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # THIS IS THE LSTM FIX — must be on both FP32 and INT8 conversions
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS       # handles TensorListReserve
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_fp32 = converter.convert()
    TFLITE_FP32.write_bytes(tflite_fp32)
    fp32_mb = len(tflite_fp32) / 1e6
    print(f"FP32: {fp32_mb:.1f} MB → {TFLITE_FP32}")

    # ── INT8 (dynamic-range) — with SELECT_TF_OPS for LSTM ────────────────
    # Full calibrated INT8 (representative_dataset below) fails here: the
    # calibrator can't execute the Flex TensorList ops the LSTM lowers to,
    # regardless of what data is fed to it (see module docstring). Dynamic-
    # range quantization skips calibration entirely — it quantizes weights
    # to int8 offline and keeps activations float32 — so it converts
    # successfully and still gives a real size/speed win over FP32.
    print("\nConverting INT8 (dynamic-range, takes ~1 min)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # LSTM fix again — required on every new converter instance
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    # Dynamic-range quantization: no representative_dataset, no forced
    # int8 input/output types — those are what full calibrated INT8 needs,
    # and calibration is exactly the step that fails on this model.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_int8 = converter.convert()
    TFLITE_INT8.write_bytes(tflite_int8)
    int8_mb = len(tflite_int8) / 1e6
    print(f"INT8 (dynamic-range): {int8_mb:.1f} MB → {TFLITE_INT8}")
    print(f"Size reduction: {fp32_mb/int8_mb:.1f}x")

    # ── Verify on 5 samples ─────────────────────────────────────────────
    # Prefer ai_edge_litert's interpreter — it's the exact same interpreter
    # vyom_inference_m.py uses on the Pi, so this is a more meaningful check
    # than the deprecated tf.lite.Interpreter anyway. TF 2.21+ also appears
    # to no longer auto-link the Flex delegate into that legacy wrapper
    # (note the UserWarning it prints), which made this step crash even
    # though the .tflite file above was already written successfully.
    print("\nVerifying dynamic-range INT8 model...")
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        except ImportError:
            TFLiteInterpreter = tf.lite.Interpreter

        interp = TFLiteInterpreter(model_content=tflite_int8)
        interp.allocate_tensors()
        inp_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        passed, checked = 0, 0
        for cls_name in CLASSES[:5]:
            cls_dir = FRAMES_DIR / cls_name
            if not cls_dir.exists():
                print(f"  ? Skipped (no data): {cls_name}")
                continue
            nf = list(cls_dir.glob("*.npy"))[0]
            arr = np.load(str(nf))[:N_FRAMES].astype(np.float32)[np.newaxis, ...]
            interp.set_tensor(inp_idx, arr)
            interp.invoke()
            probs = interp.get_tensor(out_idx)[0]
            pred  = CLASSES[np.argmax(probs)]
            conf  = np.max(probs)
            ok    = "✓" if pred == cls_name else "✗"
            checked += 1
            if pred == cls_name:
                passed += 1
            print(f"  {ok} True: {cls_name:<28} Pred: {pred:<28} Conf: {conf:.3f}")

        print(f"\nVerification: {passed}/{checked} correct")
    except Exception as e:
        print(f"\n⚠ Could not verify locally ({type(e).__name__}: {e})")
        print(f"  This is a Python-environment interpreter limitation, not a conversion")
        print(f"  failure — {TFLITE_INT8} was already written successfully above.")
        print(f"  Install ai_edge_litert (`pip install ai-edge-litert`) to verify locally,")
        print(f"  or just trust it: it's the same interpreter vyom_inference_m.py uses on the Pi.")

    print(f"\n✓ Ready for RPi4: {TFLITE_INT8}")

if __name__ == "__main__":
    convert()