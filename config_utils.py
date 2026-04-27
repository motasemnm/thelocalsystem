"""
config_utils.py
---------------
Utility helpers for runtime configuration and directory setup.
Called once at startup to ensure all required folders exist.
"""

import os
from app.config import (
    BASE_DIR, DB_PATH, MODELS_DIR,
    FACES_DIR, EMBEDDINGS_DIR, LOGS_DIR, CONFIG_DIR,
)


REQUIRED_DIRS = [
    os.path.dirname(DB_PATH),   # data/
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
    Warn if critical files are missing before the system starts.
    Returns True if all OK, False if something is missing.
    """
    issues = []

    service_account = os.path.join(
        os.path.dirname(DB_PATH).replace("data", "config"),
        "serviceAccountKey.json"
    )
    from app.config import SERVICE_ACCOUNT_PATH
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        issues.append(f"Missing Firebase credentials: {SERVICE_ACCOUNT_PATH}")

    facenet_path = os.path.join(MODELS_DIR, "facenet.tflite")
    if not os.path.exists(facenet_path):
        issues.append(f"Missing FaceNet model: {facenet_path} — run model_sync.py")

    spoof_path = os.path.join(MODELS_DIR, "anti_spoofing.tflite")
    if not os.path.exists(spoof_path):
        issues.append(f"Missing Anti-Spoofing model: {spoof_path} — run model_sync.py")

    if issues:
        print("⚠ Startup warnings:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("✔ All required files found")
    return True


def print_system_info():
    """Print a summary of the current system configuration."""
    print(f"\n{'='*40}")
    print(f"  Smart Home Auth — System Info")
    print(f"{'='*40}")
    print(f"  Base dir   : {BASE_DIR}")
    print(f"  DB path    : {DB_PATH}")
    print(f"  Models dir : {MODELS_DIR}")
    print(f"  Faces dir  : {FACES_DIR}")
    print(f"  Embeddings : {EMBEDDINGS_DIR}")
    print(f"{'='*40}\n")
