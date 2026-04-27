import os
import cv2


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise Exception(f"Failed to load Haar cascade: {CASCADE_PATH}")
    return detector


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Failed to read image: {image_path}")
    return img


def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)):
    detector = load_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
    )
    return faces


def crop_largest_face(image):
    """
    Try to detect a face with progressively relaxed parameters.
    Strict pass first (fewer false positives), then relaxed fallbacks.
    If all attempts fail, raise an exception.
    """
    attempts = [
        # (scaleFactor, minNeighbors, minSize)  — strictest first
        (1.1, 5, (60, 60)),
        (1.1, 3, (40, 40)),
        (1.05, 3, (30, 30)),
        (1.05, 2, (20, 20)),
    ]

    for scale, neighbors, min_size in attempts:
        faces = detect_faces(image, scaleFactor=scale,
                             minNeighbors=neighbors, minSize=min_size)
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest

            # Add a small padding around the detected face (10%)
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            h_img, w_img = image.shape[:2]

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            return face

    raise Exception("No face detected in image (tried all detection parameters)")


def get_cropped_face_from_path(image_path):
    image = read_image(image_path)
    face = crop_largest_face(image)
    return face


def save_cropped_face(input_path, output_path):
    face = get_cropped_face_from_path(input_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ok = cv2.imwrite(output_path, face)
    if not ok:
        raise Exception(f"Failed to save cropped face to: {output_path}")
    return output_path


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    output_path = input("Enter output path for cropped face: ").strip()
    try:
        saved = save_cropped_face(image_path, output_path)
        print(f"✔ Cropped face saved to: {saved}")
    except Exception as e:
        print(f"Error: {e}")
