"""
generate_facenet_tflite.py
--------------------------
Run this ONCE to download the official FaceNet model via deepface
and convert it to a working TFLite file.

Usage (inside your venv):
    pip install deepface
    python generate_facenet_tflite.py

Output:
    /opt/smart_home/models/facenet.tflite   ← replace your broken model with this
"""

import os
import tensorflow as tf

OUTPUT_PATH = "/opt/smart_home/models/facenet.tflite"

print("⬇ Downloading FaceNet via deepface (first run downloads weights ~90MB)...")

from deepface import DeepFace
from deepface.models.facial_recognition.Facenet import scaling

# Build the official FaceNet model (128-dim embeddings, 160x160 input)
model = DeepFace.build_model("Facenet")

# Save as Keras format first
keras_path = "/tmp/facenet.keras"
model.model.save(keras_path)
print(f"✔ Saved Keras model to: {keras_path}")

# Reload with custom scaling layer
model_keras = tf.keras.models.load_model(
    keras_path,
    custom_objects={"scaling": scaling}
)

# Convert to TFLite (float16 optimized — smaller, faster, same accuracy)
print("⚙ Converting to TFLite (float16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"✅ facenet.tflite saved to: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print()
print("Next steps:")
print("  1. Delete old embeddings:  rm -rf /opt/smart_home/embeddings/")
print("  2. Regenerate embeddings:  python -m app.embeddings")
print("  3. Test recognition:       python -m app.main")
