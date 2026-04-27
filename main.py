import os
import cv2
import numpy as np
import tempfile
import time

from app.home_sync import sync_selected_home, set_device_offline
from app.recognition import recognize_from_image
from app.db import insert_access_log, get_connection
from app.log_sync import upload_pending_logs
from app.anti_spoof import check_spoof


COOLDOWN_SECONDS = 5
FACE_CONFIRM_FRAMES = 3
RESULT_DISPLAY_SEC = 3

CAMERA_SOURCE = "/dev/video2"


def get_user_name(user_uid):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE uid = ?", (user_uid,))
        row = cursor.fetchone()
        conn.close()
        return row["name"] if row else "Unknown"
    except Exception:
        return "Unknown"


def show_result_screen(frame_shape, granted, name=None):
    h, w = frame_shape[:2]
    deadline = time.time() + RESULT_DISPLAY_SEC

    while time.time() < deadline:
        screen = np.zeros((h, w, 3), dtype="uint8")

        if granted:
            screen[:] = (34, 139, 34)

            cv2.putText(screen, "ACCESS GRANTED",
                        (w // 2 - 200, h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

            cv2.putText(screen, f"Welcome, {name}!",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

            cv2.putText(screen, "DOOR UNLOCKED",
                        (w // 2 - 150, h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
        else:
            screen[:] = (30, 30, 180)

            cv2.putText(screen, "ACCESS DENIED",
                        (w // 2 - 190, h // 2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

            cv2.putText(screen, "Unauthorized Person",
                        (w // 2 - 190, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

            cv2.putText(screen, "DOOR LOCKED",
                        (w // 2 - 130, h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)

        cv2.imshow("Smart Home Auth", screen)
        cv2.waitKey(1)


def _process_frame(frame, home_id, device_id):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    cv2.imwrite(tmp_path, frame)

    try:
        spoof = check_spoof(tmp_path)
        print(
            f"   Spoof → is_real={spoof['is_real']} "
            f"(real={spoof.get('real_score', 0):.3f} / fake={spoof.get('fake_score', 0):.3f})"
        )

        if not spoof["is_real"]:
            print("   🚨 Spoof detected! Access denied")

            show_result_screen(frame.shape, granted=False)

            insert_access_log(
                home_id=home_id,
                device_id=device_id,
                user_uid=None,
                result="denied",
                action="spoof_detected",
                confidence=spoof["score"],
                spoof_result="fake",
            )

            upload_pending_logs()
            return

        result = recognize_from_image(home_id, tmp_path)

        if result["matched"]:
            name = get_user_name(result["user_uid"])

            print(
                f"   ✅ Access GRANTED — {name} "
                f"(distance: {result['distance']:.4f})"
            )

            show_result_screen(frame.shape, granted=True, name=name)

            insert_access_log(
                home_id=home_id,
                device_id=device_id,
                user_uid=result["user_uid"],
                result="granted",
                action="door_unlock",
                confidence=result["distance"],
                spoof_result="real",
            )

        else:
            print(
                f"   ❌ Access DENIED — unauthorized "
                f"(distance: {result['distance']:.4f})"
            )

            show_result_screen(frame.shape, granted=False)

            insert_access_log(
                home_id=home_id,
                device_id=device_id,
                user_uid=None,
                result="denied",
                action="door_locked",
                confidence=result["distance"],
                spoof_result="real",
            )

        upload_pending_logs()

    except Exception as e:
        print(f"❌ Processing error: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_camera(home_id, device_id):
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise Exception(
            f"Could not open camera source: {CAMERA_SOURCE}. "
            "Check that your camera is connected."
        )

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise Exception("Could not load OpenCV face detector.")

    print("\n🎥 Camera is running — press Q to quit\n")

    last_decision_time = 0
    confirm_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠ Failed to read frame — retrying...")
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time()
        in_cooldown = (now - last_decision_time) < COOLDOWN_SECONDS

        if in_cooldown:
            remaining = int(COOLDOWN_SECONDS - (now - last_decision_time))
            cv2.putText(
                frame,
                f"Please wait... {remaining}s",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2
            )
            confirm_count = 0

        elif face_detected:
            confirm_count += 1
            cv2.putText(
                frame,
                f"Hold still... ({confirm_count}/{FACE_CONFIRM_FRAMES})",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:
            confirm_count = 0
            cv2.putText(
                frame,
                "Scanning for face...",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 200),
                2
            )

        cv2.imshow("Smart Home Auth", frame)

        if face_detected and not in_cooldown and confirm_count >= FACE_CONFIRM_FRAMES:
            confirm_count = 0
            last_decision_time = time.time()

            print("\n📸 Face confirmed — processing...")
            _process_frame(frame.copy(), home_id, device_id)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n🛑 Shutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    home_id = input("Enter Home ID: ").strip()
    device_id = input("Enter Device ID: ").strip()

    try:
        print("\n=== Syncing home ===")
        sync_selected_home(home_id, device_id)

        run_camera(home_id, device_id)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            set_device_offline(home_id, device_id)
            print("✔ Device set to offline in Firebase")
        except Exception as e:
            print(f"Failed to set device offline: {e}")


if __name__ == "__main__":
    main()
