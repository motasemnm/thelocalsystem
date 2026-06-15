import os
import numpy as np
import cv2
import tensorflow as tf

from app.face_utils import get_cropped_face_from_path
from app.db import get_connection, insert_face_image
from app.config import EMBEDDINGS_DIR, MODELS_DIR, FACES_DIR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
    model_path = os.path.join(MODELS_DIR, "facenet.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path):
    face = get_cropped_face_from_path(image_path)
    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")

    mean = np.mean(face)
    std = np.std(face)
    std = max(std, 1.0 / np.sqrt(face.size))
    face = (face - mean) / std

    return np.expand_dims(face, axis=0)


def generate_embedding(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]["index"])[0]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def home_exists(home_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM homes WHERE id = ?", (home_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def user_exists(user_uid):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT uid FROM users WHERE uid = ?", (user_uid,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def home_user_exists(home_id, user_uid):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM homes WHERE id = ?", (home_id,))
    home = cursor.fetchone()

    cursor.execute("SELECT uid FROM users WHERE uid = ?", (user_uid,))
    user = cursor.fetchone()

    conn.close()

    return home is not None and user is not None


def process_existing_face_rows(home_id=None):
    conn = get_connection()
    cursor = conn.cursor()

    if home_id:
        cursor.execute("""
            SELECT id, home_id, user_uid, image_name, local_path
            FROM face_images
            WHERE home_id = ?
        """, (home_id,))
    else:
        cursor.execute("""
            SELECT id, home_id, user_uid, image_name, local_path
            FROM face_images
        """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def scan_faces_from_disk(home_id=None):
    scanned = []

    if not os.path.exists(FACES_DIR):
        return scanned

    home_folders = []

    if home_id:
        home_folder = f"home_{home_id}"
        home_path = os.path.join(FACES_DIR, home_folder)

        if os.path.isdir(home_path):
            home_folders.append(home_folder)
    else:
        home_folders = os.listdir(FACES_DIR)

    for home_folder in home_folders:
        home_path = os.path.join(FACES_DIR, home_folder)

        if not os.path.isdir(home_path) or not home_folder.startswith("home_"):
            continue

        current_home_id = home_folder.replace("home_", "", 1)

        if home_id and current_home_id != home_id:
            continue

        if not home_exists(current_home_id):
            continue

        for user_folder in os.listdir(home_path):
            user_path = os.path.join(home_path, user_folder)

            if not os.path.isdir(user_path) or not user_folder.startswith("user_"):
                continue

            user_uid = user_folder.replace("user_", "", 1)

            if not user_exists(user_uid):
                continue

            for file_name in os.listdir(user_path):
                image_path = os.path.join(user_path, file_name)

                if not os.path.isfile(image_path):
                    continue

                scanned.append({
                    "home_id": current_home_id,
                    "user_uid": user_uid,
                    "image_name": file_name,
                    "local_path": image_path,
                })

    return scanned


def process_all_faces(home_id=None):
    existing_rows = process_existing_face_rows(home_id)
    disk_faces = scan_faces_from_disk(home_id)

    all_faces = []

    for row in existing_rows:
        item = {
            "id": row["id"],
            "home_id": row["home_id"],
            "user_uid": row["user_uid"],
            "image_name": row["image_name"],
            "local_path": row["local_path"],
        }

        if not home_user_exists(item["home_id"], item["user_uid"]):
            continue

        if not os.path.exists(item["local_path"]):
            continue

        all_faces.append(item)

    seen_keys = {
        (item["home_id"], item["user_uid"], item["image_name"])
        for item in all_faces
    }

    for item in disk_faces:
        key = (item["home_id"], item["user_uid"], item["image_name"])

        if key in seen_keys:
            continue

        if not home_user_exists(item["home_id"], item["user_uid"]):
            continue

        insert_face_image(
            home_id=item["home_id"],
            user_uid=item["user_uid"],
            image_name=item["image_name"],
            local_path=item["local_path"],
            cloud_path=None,
            download_url=None,
            embedding_path=None,
            status="active",
            created_at=None,
            updated_at=None,
        )

        all_faces.append(item)
        seen_keys.add(key)

    if not all_faces:
        print("No face images found")
        return

    interpreter = load_model()

    print("=== Generating embeddings ===")

    success_count = 0
    fail_count = 0

    for item in all_faces:
        current_home_id = item["home_id"]
        user_uid = item["user_uid"]
        image_name = item["image_name"]
        image_path = item["local_path"]

        try:
            if not home_user_exists(current_home_id, user_uid):
                fail_count += 1
                continue

            if not os.path.exists(image_path):
                fail_count += 1
                continue

            input_data = preprocess_image(image_path)
            embedding = generate_embedding(interpreter, input_data)

            embed_dir = os.path.join(
                EMBEDDINGS_DIR,
                f"home_{current_home_id}",
                f"user_{user_uid}",
            )

            ensure_dir(embed_dir)

            embedding_path = os.path.join(embed_dir, f"{image_name}.npy")
            np.save(embedding_path, embedding)

            insert_face_image(
                home_id=current_home_id,
                user_uid=user_uid,
                image_name=image_name,
                local_path=image_path,
                cloud_path=None,
                download_url=None,
                embedding_path=embedding_path,
                status="active",
                created_at=None,
                updated_at=None,
            )

            print(f"Embedding created: {embedding_path}")
            success_count += 1

        except Exception as e:
            print(f"Failed to create embedding for {image_path}: {e}")
            fail_count += 1

    print(f"Embedding generation complete. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    selected_home_id = input("Enter home ID or leave empty for all: ").strip()
    if selected_home_id:
        process_all_faces(selected_home_id)
    else:
        process_all_faces()
