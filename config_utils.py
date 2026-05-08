import json
import os

from app.config import (
    BASE_DIR,
    DB_PATH,
    MODELS_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    DEVICE_CONFIG_PATH,
)


REQUIRED_DIRS = [
    os.path.dirname(DB_PATH),
    MODELS_DIR,
    FACES_DIR,
    EMBEDDINGS_DIR,
    LOGS_DIR,
    CONFIG_DIR,
]


def ensure_all_dirs():
    """Create all required runtime directories if they don't exist."""
    for path in REQUIRED_DIRS:
        os.makedirs(path, exist_ok=True)

    print("✔ All required directories verified")


def check_required_files():
    """
    Check local runtime files.

    We no longer require serviceAccountKey.json because Firebase Admin SDK
    should not be stored on the Ubuntu device anymore.
    """

    issues = []

    if not os.path.exists(DEVICE_CONFIG_PATH):
        issues.append(
            f"Missing device config: {DEVICE_CONFIG_PATH} "
            "— create device_config.json with homeId, deviceId, and deviceSecret."
        )

    facenet_path = os.path.join(MODELS_DIR, "facenet.tflite")
    if not os.path.exists(facenet_path):
        issues.append(
            f"Missing FaceNet model: {facenet_path} — run model_sync.py"
        )

    spoof_path = os.path.join(MODELS_DIR, "anti_spoofing.tflite")
    if not os.path.exists(spoof_path):
        issues.append(
            f"Missing Anti-Spoofing model: {spoof_path} — run model_sync.py"
        )

    if issues:
        print("⚠ Startup warnings:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✔ All required files found")
    return True


def load_device_config():
    """Load homeId, deviceId, and deviceSecret from device_config.json."""
    if not os.path.exists(DEVICE_CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing device config file: {DEVICE_CONFIG_PATH}"
        )

    with open(DEVICE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    home_id = str(data.get("homeId", "")).strip()
    device_id = str(data.get("deviceId", "")).strip()
    device_secret = str(data.get("deviceSecret", "")).strip()

    if not home_id or not device_id or not device_secret:
        raise ValueError(
            "device_config.json must contain homeId, deviceId, and deviceSecret."
        )

    return {
        "homeId": home_id,
        "deviceId": device_id,
        "deviceSecret": device_secret,
    }


def print_system_info():
    """Print a summary of the current system configuration."""
    print(f"\n{'=' * 40}")
    print("  Smart Home Auth — System Info")
    print(f"{'=' * 40}")
    print(f"  Base dir   : {BASE_DIR}")
    print(f"  DB path    : {DB_PATH}")
    print(f"  Models dir : {MODELS_DIR}")
    print(f"  Faces dir  : {FACES_DIR}")
    print(f"  Embeddings : {EMBEDDINGS_DIR}")
    print(f"  Config dir : {CONFIG_DIR}")
    print(f"{'=' * 40}\n")

