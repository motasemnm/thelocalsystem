import os
import numpy as np
import cv2
import tensorflow as tf
from app.face_utils import get_cropped_face_from_path
from app.config import MODELS_DIR, EMBEDDINGS_DIR


# Distance below this = same person.
# With L2-normalized embeddings + standardized input, 0.7 is a good
# starting point. Lower = stricter (fewer false accepts).
MATCH_THRESHOLD = 0.7


def load_facenet():
    model_path = os.path.join(MODELS_DIR, "facenet.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path):
    """
    Must match exactly the preprocessing used in embeddings.py.
    FaceNet expects per-image standardization (zero mean, unit variance),
    NOT simple /255 scaling.
    """
    face = get_cropped_face_from_path(image_path)
    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32")

    mean = np.mean(face)
    std  = np.std(face)
    std  = max(std, 1.0 / np.sqrt(face.size))
    face = (face - mean) / std

    return np.expand_dims(face, axis=0)


def generate_embedding(interpreter, input_data):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    embedding = interpreter.get_tensor(output_details[0]["index"])[0]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def load_embeddings_for_home(home_id):
    home_id  = home_id.strip()
    home_dir = os.path.join(EMBEDDINGS_DIR, f"home_{home_id}")

    if not os.path.exists(home_dir):
        print(f"[DEBUG] Home embeddings directory not found: {home_dir}")
        return []

    known = []

    user_dirs = [
        d for d in os.listdir(home_dir)
        if os.path.isdir(os.path.join(home_dir, d)) and d.startswith("user_")
    ]

    print(f"[DEBUG] Found user embedding folders: {len(user_dirs)}")

    for user_dir in user_dirs:
        user_uid  = user_dir.replace("user_", "", 1)
        user_path = os.path.join(home_dir, user_dir)

        for file_name in os.listdir(user_path):
            if not file_name.endswith(".npy"):
                continue

            emb_path = os.path.join(user_path, file_name)

            try:
                emb = np.load(emb_path)
                known.append({
                    "user_uid":       user_uid,
                    "embedding":      emb,
                    "embedding_path": emb_path,
                })
            except Exception as e:
                print(f"[DEBUG] Failed to load embedding {emb_path}: {e}")

    print(f"[DEBUG] Total embeddings loaded: {len(known)}")
    return known


def compare_embeddings(test_embedding, known_embeddings):
    if not known_embeddings:
        return {
            "matched":       False,
            "user_uid":      None,
            "distance":      float("inf"),
            "all_distances": [],
        }

    all_distances  = []
    best_user_uid  = None
    best_distance  = float("inf")

    # Group embeddings by user and take the MINIMUM distance per user.
    # A user may have multiple face images — we match if ANY is close enough.
    user_best = {}

    for item in known_embeddings:
        distance = float(np.linalg.norm(test_embedding - item["embedding"]))
        uid      = item["user_uid"]

        all_distances.append({
            "user_uid":       uid,
            "distance":       distance,
            "embedding_path": item["embedding_path"],
        })

        if uid not in user_best or distance < user_best[uid]:
            user_best[uid] = distance

    print("[DEBUG] Best distance per user:")
    for uid, dist in user_best.items():
        print(f"  user_uid={uid} | best_distance={dist:.6f}")

    best_user_uid = min(user_best, key=user_best.get)
    best_distance = user_best[best_user_uid]

    if best_distance <= MATCH_THRESHOLD:
        return {
            "matched":       True,
            "user_uid":      best_user_uid,
            "distance":      best_distance,
            "all_distances": all_distances,
        }

    return {
        "matched":       False,
        "user_uid":      None,
        "distance":      best_distance,
        "all_distances": all_distances,
    }


def recognize_from_image(home_id, image_path):
    known_embeddings = load_embeddings_for_home(home_id)

    if not known_embeddings:
        raise Exception(f"No embeddings found for home: {home_id}")

    interpreter    = load_facenet()
    input_data     = preprocess_image(image_path)
    test_embedding = generate_embedding(interpreter, input_data)

    result = compare_embeddings(test_embedding, known_embeddings)
    return result


if __name__ == "__main__":
    home_id    = input("Enter Home ID: ").strip()
    image_path = input("Enter image path to test: ").strip()

    try:
        result = recognize_from_image(home_id, image_path)
        print("Recognition result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
