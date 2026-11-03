import tensorflow as tf
import tf2onnx

MODEL_PATH = "models/vyom_mobilenet_lstm.h5"
ONNX_PATH = "models/vyom.onnx"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

spec = (
    tf.TensorSpec((1,45,160,160,3), tf.float32, name="input"),
)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=ONNX_PATH
)

print("Saved:", ONNX_PATH)