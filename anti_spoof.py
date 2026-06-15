import os
import numpy as np
import cv2
import tensorflow as tf

from app.config import MODELS_DIR
from app.face_utils import get_cropped_face_from_path

MODEL_PATH = os.path.join(MODELS_DIR, "anti_spoofing.tflite")
REAL_THRESHOLD = 0.90


def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(image_path, input_shape):
    try:
        img = get_cropped_face_from_path(image_path)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Failed to read image: {image_path}")

    height = input_shape[1]
    width = input_shape[2]

    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def interpret_output(output_flat):
    if len(output_flat) == 1:
        score = float(output_flat[0])
        is_real = score >= REAL_THRESHOLD

        return {
            "is_real": is_real,
            "score": score,
            "fake_score": 1.0 - score,
            "real_score": score,
        }

    if len(output_flat) >= 2:
        fake_score = float(output_flat[0])
        real_score = float(output_flat[1])

        exp_fake = np.exp(fake_score - max(fake_score, real_score))
        exp_real = np.exp(real_score - max(fake_score, real_score))
        total = exp_fake + exp_real

        fake_prob = exp_fake / total
        real_prob = exp_real / total

        is_real = real_prob >= REAL_THRESHOLD and real_prob > fake_prob

        return {
            "is_real": is_real,
            "score": float(real_prob),
            "fake_score": float(fake_prob),
            "real_score": float(real_prob),
        }

    raise Exception(f"Unexpected anti-spoof output shape: {output_flat.shape}")


def check_spoof(image_path):
    interpreter = load_model()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_data = preprocess(image_path, input_shape)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    output_flat = output.flatten()

    result = interpret_output(output_flat)
    result["raw_output"] = output.tolist()
    result["input_shape"] = input_shape.tolist()

    return result
