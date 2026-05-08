def initialize_firebase():
    raise RuntimeError(
        "Firebase Admin SDK is disabled. "
        "Use device_api.py (Cloud Functions) instead."
    )


def get_firestore():
    raise RuntimeError(
        "Direct Firestore access is disabled. "
        "Use device_api.sync_home_data() instead."
    )


def get_bucket():
    raise RuntimeError(
        "Direct Storage access is disabled. "
        "Use device_api.download_file_from_storage() instead."
    )
