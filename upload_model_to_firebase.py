"""
upload_model_to_firebase.py
---------------------------
After generating a new facenet.tflite, run this script to upload it
to Firebase Storage so all other devices get the new model
when they run model_sync.py.

Usage (inside your venv, from /opt/smart_home):
    python app/upload_model_to_firebase.py
"""

import os
from app.firebase_client import get_bucket
from app.config import MODELS_DIR


def upload_model(local_filename, cloud_path):
    local_path = os.path.join(MODELS_DIR, local_filename)

    if not os.path.exists(local_path):
        print(f"❌ Local model not found: {local_path}")
        print("   Run generate_facenet_tflite.py first.")
        return

    bucket = get_bucket()
    blob = bucket.blob(cloud_path)

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"⬆ Uploading {local_filename} ({size_mb:.1f} MB) → {cloud_path} ...")

    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded successfully: {cloud_path}")


if __name__ == "__main__":
    # Upload FaceNet — replaces the broken model in Firebase Storage
    upload_model(
        local_filename="facenet.tflite",
        cloud_path="models/facenet/facenet.tflite",
    )
