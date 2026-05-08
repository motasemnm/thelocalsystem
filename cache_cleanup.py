import os
import time

from app.config import (
    FACES_DIR,
    EMBEDDINGS_DIR,
    LOGS_DIR,
    MODELS_DIR,
)
from app.db import get_connection


CACHE_SECONDS = 60 * 60
MODEL_CACHE_SECONDS = 60 * 60


def _delete_old_files(folder, max_age_seconds):
    if not os.path.exists(folder):
        return

    now = time.time()

    for root, dirs, files in os.walk(folder):
        for file_name in files:
            path = os.path.join(root, file_name)

            try:
                age = now - os.path.getmtime(path)

                if age > max_age_seconds:
                    os.remove(path)
                    print(f" Deleted old cache file: {path}")

            except Exception as e:
                print(f" Could not delete {path}: {e}")


def cleanup_old_local_cache():
    print("\n=== Cleaning local cache ===")

    _delete_old_files(FACES_DIR, CACHE_SECONDS)
    _delete_old_files(EMBEDDINGS_DIR, CACHE_SECONDS)
    _delete_old_files(LOGS_DIR, CACHE_SECONDS)


    conn = get_connection()
    cursor = conn.cursor()


    cursor.execute("""
        DELETE FROM face_images
        WHERE updated_at IS NOT NULL
        AND datetime(updated_at) < datetime('now', '-1 hour')
    """)

    conn.commit()
    conn.close()

    print(" Cache cleanup complete")
