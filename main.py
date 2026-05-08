import os
import cv2
import numpy as np
import tempfile
import time

from app.config_utils import load_device_config
from app.home_sync import sync_selected_home, set_device_offline
from app.recognition import recognize_from_image
from app.db import insert_access_log, get_connection
from app.log_sync import upload_pending_logs
from app.sync_logs import process_sync_queue
from app.anti_spoof import check_spoof
from app.embeddings import process_all_faces
from app.model_sync import download_models
from app.cache_cleanup import cleanup_old_local_cache
from app.security_alerts import check_tampering_and_alert


COOLDOWN_SECONDS = 6
FACE_CONFIRM_FRAMES = 4
RESULT_DISPLAY_SEC = 4
CAMERA_SOURCE = "/dev/video0"
REFRESH_SECONDS = 60
BLINK_SECONDS = 5


def get_user_name(user_uid):
    try:
        if user_uid is None:
            return "Unknown"

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE uid = ?", (user_uid,))
        row = cursor.fetchone()
        conn.close()

        return row["name"] if row else "Unknown"
    except Exception:
        return "Unknown"


def check_all_local_blockchains(device_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT home_id
            FROM access_logs
            WHERE home_id IS NOT NULL
            AND home_id != ''
        """)

        homes = [row["home_id"] for row in cursor.fetchall()]
        conn.close()

        if not homes:
            print("No local access logs found to verify.")
            return

        for local_home_id in homes:
            print(f"\n=== Checking blockchain for local home {local_home_id} ===")
            check_tampering_and_alert(local_home_id, device_id)

    except Exception as e:
        print(f"Blockchain check error, system will continue: {e}")


def detect_blink_simple(cap):
    print("Checking blink...")

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    if eye_cascade.empty():
        print("Could not load eye detector")
        return False

    blink_detected = False
    eyes_were_open = False
    eyes_closed_after_open = False
    start_time = time.time()

    while time.time() - start_time < BLINK_SECONDS:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(25, 25),
        )

        eyes_now = len(eyes)

        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if eyes_now >= 1:
            eyes_were_open = True

            if eyes_closed_after_open:
                blink_detected = True
                cv2.putText(
                    frame,
                    "Blink detected!",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Smart Home Auth", frame)
                cv2.waitKey(500)
                break

        elif eyes_were_open and eyes_now == 0:
            eyes_closed_after_open = True

        cv2.putText(
            frame,
            "Blink now...",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Eyes detected: {eyes_now}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Smart Home Auth", frame)
        cv2.waitKey(1)

    return blink_detected


def show_result_screen(frame_shape, granted, name=None):
    h, w = frame_shape[:2]
    deadline = time.time() + RESULT_DISPLAY_SEC

    while time.time() < deadline:
        screen = np.zeros((h, w, 3), dtype="uint8")

        if granted:
            screen[:] = (34, 139, 34)

            cv2.putText(
                screen,
                "ACCESS GRANTED",
                (w // 2 - 200, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3,
            )

            cv2.putText(
                screen,
                f"Welcome, {name}!",
                (w // 2 - 160, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                screen,
                "DOOR UNLOCKED",
                (w // 2 - 150, h // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 0),
                2,
            )

        else:
            screen[:] = (30, 30, 180)

            cv2.putText(
                screen,
                "ACCESS DENIED",
                (w // 2 - 190, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3,
            )

            cv2.putText(
                screen,
                "Unauthorized / No Liveness",
                (w // 2 - 230, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                screen,
                "DOOR LOCKED",
                (w // 2 - 130, h // 2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 0),
                2,
            )

        cv2.imshow("Smart Home Auth", screen)
        cv2.waitKey(1)


def refresh_all_cloud_cache(home_id, device_id):
    try:
        print("\n=== Auto refresh: cache + cloud data + AI models ===")

        # Check blockchain first.
        print("\n=== Checking blockchain/log integrity ===")
        check_all_local_blockchains(device_id)

        cleanup_old_local_cache()
        download_models()
        sync_selected_home(home_id, device_id)
        process_all_faces()
        upload_pending_logs()
        process_sync_queue()

        print("Auto refresh complete")

    except Exception as e:
        print(f"Auto refresh skipped/offline: {e}")


def _process_frame(frame, home_id, device_id, cap):
    blink_ok = detect_blink_simple(cap)

    if not blink_ok:
        print("Blink not detected → spoof / no liveness")

        show_result_screen(frame.shape, granted=False)

        insert_access_log(
            home_id=home_id,
            device_id=device_id,
            user_uid=None,
            result="denied",
            action="no_liveness",
            confidence=0,
            spoof_result="fake",
        )

        upload_pending_logs()
        return

    print("Blink detected → continue")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    cv2.imwrite(tmp_path, frame)

    try:
        spoof_result = "real"

        try:
            spoof = check_spoof(tmp_path)

            print(
                f"   Anti-spoof model score only → is_real={spoof['is_real']} "
                f"(real={spoof.get('real_score', 0):.3f} / "
                f"fake={spoof.get('fake_score', 0):.3f})"
            )

            spoof_result = "real" if spoof.get("is_real") else "model_flagged_fake"

        except Exception as e:
            print(f"Anti-spoof model skipped/error: {e}")
            spoof_result = "blink_verified_model_error"

        result = recognize_from_image(home_id, tmp_path)

        if result["matched"]:
            name = get_user_name(result["user_uid"])

            show_result_screen(frame.shape, granted=True, name=name)

            insert_access_log(
                home_id=home_id,
                device_id=device_id,
                user_uid=result["user_uid"],
                result="granted",
                action="door_unlock",
                confidence=result["distance"],
                spoof_result=spoof_result,
            )

        else:
            show_result_screen(frame.shape, granted=False)

            insert_access_log(
                home_id=home_id,
                device_id=device_id,
                user_uid=None,
                result="denied",
                action="door_locked",
                confidence=result["distance"],
                spoof_result=spoof_result,
            )

        upload_pending_logs()

    except Exception as e:
        print(f"Processing error: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_camera(home_id, device_id):
    cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise Exception(f"Could not open camera source: {CAMERA_SOURCE}")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise Exception("Could not load OpenCV face detector.")

    print("\nCamera is running — press Q to quit\n")

    last_decision_time = 0
    confirm_count = 0
    last_refresh_time = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame — retrying...")
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time()

        if now - last_refresh_time >= REFRESH_SECONDS:
            refresh_all_cloud_cache(home_id, device_id)
            last_refresh_time = now

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
                2,
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
                2,
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
                2,
            )

        cv2.imshow("Smart Home Auth", frame)

        if face_detected and not in_cooldown and confirm_count >= FACE_CONFIRM_FRAMES:
            confirm_count = 0
            last_decision_time = time.time()

            print("\nFace confirmed — blink challenge starting...")
            _process_frame(frame.copy(), home_id, device_id, cap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nShutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    device_config = load_device_config()
    home_id = device_config["homeId"]
    device_id = device_config["deviceId"]

    try:
        print("\n=== Startup sync ===")
        refresh_all_cloud_cache(home_id, device_id)
        run_camera(home_id, device_id)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            set_device_offline(home_id, device_id)
            print("Device set to offline via API")
        except Exception as e:
            print(f"Failed to set device offline: {e}")


if __name__ == "__main__":
    main()
