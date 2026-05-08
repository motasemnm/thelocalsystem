import os
from datetime import datetime, timezone

from app.device_api import download_file_from_storage
from app.config import MODELS_DIR
from app.db import insert_model


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_models():
    ensure_dir(MODELS_DIR)

    print("=== Downloading models via API ===")

    now = datetime.now(timezone.utc).isoformat()

    # -------------------------
    # FaceNet
    # -------------------------
    facenet_path = os.path.join(MODELS_DIR, "facenet.tflite")
    facenet_cloud_path = "models/facenet/facenet.tflite"

    if not os.path.exists(facenet_path):
        try:
            download_file_from_storage(facenet_cloud_path, facenet_path)
            print(f"✔ Downloaded FaceNet: {facenet_path}")
        except Exception as e:
            print(f" Failed to download FaceNet: {e}")
    else:
        print("ℹ FaceNet already exists")

    insert_model(
        model_name="facenet",
        version="1.0",
        local_path=facenet_path,
        cloud_path=facenet_cloud_path,
        active=1,
        downloaded_at=now,
    )

    # -------------------------
    # Anti Spoofing
    # -------------------------
    spoof_path = os.path.join(MODELS_DIR, "anti_spoofing.tflite")
    spoof_cloud_path = "models/anti-spoofing/FaceAntiSpoofing.tflite"

    if not os.path.exists(spoof_path):
        try:
            download_file_from_storage(spoof_cloud_path, spoof_path)
            print(f" Downloaded Anti-Spoofing: {spoof_path}")
        except Exception as e:
            print(f" Failed to download Anti-Spoofing: {e}")
    else:
        print("ℹ Anti-Spoofing model already exists")

    insert_model(
        model_name="anti_spoofing",
        version="1.0",
        local_path=spoof_path,
        cloud_path=spoof_cloud_path,
        active=1,
        downloaded_at=now,
    )

    print(" Model sync complete")


if __name__ == "__main__":
    download_models()
