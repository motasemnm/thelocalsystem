import os
from datetime import datetime, timezone
from app.firebase_client import get_bucket
from app.config import MODELS_DIR
from app.db import insert_model


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_models():
    bucket = get_bucket()
    ensure_dir(MODELS_DIR)

    print("=== Downloading models ===")

    now = datetime.now(timezone.utc).isoformat()

    # -------------------------
    # FaceNet
    # -------------------------
    facenet_blob = bucket.blob("models/facenet/facenet.tflite")
    facenet_path = os.path.join(MODELS_DIR, "facenet.tflite")

    if not os.path.exists(facenet_path):
        facenet_blob.download_to_filename(facenet_path)
        print(f"✔ Downloaded FaceNet: {facenet_path}")
    else:
        print(f"ℹ FaceNet already exists")

    # Register / update in local DB so the system can track active models
    insert_model(
        model_name="facenet",
        version="1.0",
        local_path=facenet_path,
        cloud_path="models/facenet/facenet.tflite",
        active=1,
        downloaded_at=now,
    )

    # -------------------------
    # Anti Spoofing
    # -------------------------
    spoof_blob = bucket.blob("models/anti-spoofing/FaceAntiSpoofing.tflite")
    spoof_path = os.path.join(MODELS_DIR, "anti_spoofing.tflite")

    if not os.path.exists(spoof_path):
        spoof_blob.download_to_filename(spoof_path)
        print(f"✔ Downloaded Anti-Spoofing: {spoof_path}")
    else:
        print(f"ℹ Anti-Spoofing model already exists")

    insert_model(
        model_name="anti_spoofing",
        version="1.0",
        local_path=spoof_path,
        cloud_path="models/anti-spoofing/FaceAntiSpoofing.tflite",
        active=1,
        downloaded_at=now,
    )

    print("✅ Model sync complete")


if __name__ == "__main__":
    download_models()
