import os
import cv2
import csv
import threading
import time
from datetime import datetime

from app.anti_spoof import check_spoof


OUTPUT_FILE = "recognition_accuracy_camera_antispoof_results.csv"
CAMERA_INDEX = 0
CAPTURE_DIR = "/opt/smart_home/data/test_captures"


class CameraStream:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


SCENARIOS = {
    "1": {
        "name": "Registered user with blink",
        "expected_access": "granted",
        "expected_spoof": "real"
    },
    "2": {
        "name": "Registered user without blink",
        "expected_access": "denied",
        "expected_spoof": "real"
    },
    "3": {
        "name": "Unknown user",
        "expected_access": "denied",
        "expected_spoof": "real"
    },
    "4": {
        "name": "Photo / spoof attempt",
        "expected_access": "denied",
        "expected_spoof": "fake"
    }
}


def choose_scenario():
    print("\nChoose test scenario:")

    for key, value in SCENARIOS.items():
        print(
            f"{key}. {value['name']} | "
            f"Expected access: {value['expected_access']} | "
            f"Expected spoof: {value['expected_spoof']}"
        )

    while True:
        choice = input("Enter scenario number: ").strip()

        if choice in SCENARIOS:
            return SCENARIOS[choice]

        print("Invalid choice. Choose 1, 2, 3, or 4.")


def ensure_capture_dir():
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def save_frame(frame, attempt_number):
    ensure_capture_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(
        CAPTURE_DIR,
        f"attempt_{attempt_number}_{timestamp}.jpg"
    )

    ok = cv2.imwrite(image_path, frame)

    if not ok:
        raise Exception("Failed to save captured frame")

    return image_path


def open_camera_capture_and_test_antispoof(scenario, attempt_number):
    stream = CameraStream(CAMERA_INDEX)

    actual_access = None

    spoof_result = None
    real_score = None
    fake_score = None
    captured_image_path = None
    anti_spoof_error = None

    print("\nCamera opened.")
    print("Press C to capture frame and run anti-spoofing model.")
    print("After anti-spoofing result appears:")
    print("Press G if the final system result is GRANTED.")
    print("Press D if the final system result is DENIED.")
    print("Press Q to skip this attempt.")

    while True:
        frame = stream.read()

        if frame is None:
            time.sleep(0.01)
            continue

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Scenario: {scenario['name']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Expected Access: {scenario['expected_access']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Expected Spoof: {scenario['expected_spoof']}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "C = AntiSpoof | G = Granted | D = Denied | Q = Quit",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        if spoof_result is not None:
            cv2.putText(
                display_frame,
                f"Anti-spoof result: {spoof_result}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Real score: {real_score:.4f} | Fake score: {fake_score:.4f}",
                (20, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if anti_spoof_error is not None:
            cv2.putText(
                display_frame,
                f"Anti-spoof error: {anti_spoof_error}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        cv2.imshow("Recognition Accuracy + Anti-Spoof Test", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            try:
                captured_image_path = save_frame(frame, attempt_number)

                result = check_spoof(captured_image_path)

                if result["is_real"]:
                    spoof_result = "real"
                else:
                    spoof_result = "fake"

                real_score = float(result.get("real_score", 0.0))
                fake_score = float(result.get("fake_score", 0.0))

                anti_spoof_error = None

                print("\nAnti-spoofing result:")
                print(f"Image path: {captured_image_path}")
                print(f"Result: {spoof_result}")
                print(f"Real score: {real_score:.4f}")
                print(f"Fake score: {fake_score:.4f}")

            except Exception as e:
                anti_spoof_error = str(e)
                print(f"\nAnti-spoofing error: {anti_spoof_error}")

        elif key == ord("g"):
            actual_access = "granted"
            break

        elif key == ord("d"):
            actual_access = "denied"
            break

        elif key == ord("q"):
            actual_access = "quit"
            break

    stream.stop()
    cv2.destroyAllWindows()

    return {
        "actual_access": actual_access,
        "spoof_result": spoof_result,
        "real_score": real_score,
        "fake_score": fake_score,
        "captured_image_path": captured_image_path,
        "anti_spoof_error": anti_spoof_error
    }


def save_results(results):
    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "attempt",
                "scenario",
                "expected_access",
                "actual_access",
                "access_correct",
                "expected_spoof",
                "actual_spoof",
                "spoof_correct",
                "real_score",
                "fake_score",
                "captured_image_path",
                "anti_spoof_error",
                "timestamp"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


def calculate_access_accuracy(results):
    if not results:
        return 0

    correct = sum(1 for r in results if r["access_correct"])
    return (correct / len(results)) * 100


def calculate_spoof_accuracy(results):
    valid_spoof_tests = [
        r for r in results
        if r["actual_spoof"] in ["real", "fake"]
    ]

    if not valid_spoof_tests:
        return 0, 0, 0

    correct = sum(1 for r in valid_spoof_tests if r["spoof_correct"])
    total = len(valid_spoof_tests)

    return (correct / total) * 100, correct, total


def calculate_false_acceptance_rate(results):
    attack_scenarios = [
        "Unknown user",
        "Photo / spoof attempt"
    ]

    attack_attempts = [
        r for r in results
        if r["scenario"] in attack_scenarios
    ]

    false_accepts = [
        r for r in attack_attempts
        if r["actual_access"] == "granted"
    ]

    if not attack_attempts:
        return 0, 0, 0

    far = (len(false_accepts) / len(attack_attempts)) * 100
    return far, len(false_accepts), len(attack_attempts)


def calculate_false_rejection_rate(results):
    valid_attempts = [
        r for r in results
        if r["scenario"] == "Registered user with blink"
    ]

    false_rejections = [
        r for r in valid_attempts
        if r["actual_access"] == "denied"
    ]

    if not valid_attempts:
        return 0, 0, 0

    frr = (len(false_rejections) / len(valid_attempts)) * 100
    return frr, len(false_rejections), len(valid_attempts)


def print_summary(results):
    print("\n========== Recognition + Anti-Spoof Summary ==========")

    total = len(results)
    access_correct = sum(1 for r in results if r["access_correct"])
    access_wrong = total - access_correct

    access_accuracy = calculate_access_accuracy(results)
    spoof_accuracy, spoof_correct, spoof_total = calculate_spoof_accuracy(results)

    far, false_accepts, attack_attempts = calculate_false_acceptance_rate(results)
    frr, false_rejections, valid_attempts = calculate_false_rejection_rate(results)

    print(f"Total Attempts: {total}")
    print(f"Correct Access Decisions: {access_correct}")
    print(f"Wrong Access Decisions: {access_wrong}")
    print(f"Access Accuracy: {access_accuracy:.2f}%")

    print("\nAnti-Spoofing Metrics")
    print(f"Anti-Spoof Accuracy: {spoof_accuracy:.2f}%")
    print(f"Correct Anti-Spoof Results: {spoof_correct}/{spoof_total}")

    print("\nSecurity Metrics")
    print(f"False Acceptance Rate: {far:.2f}%")
    print(f"False Acceptances: {false_accepts}/{attack_attempts}")

    print(f"False Rejection Rate: {frr:.2f}%")
    print(f"False Rejections: {false_rejections}/{valid_attempts}")

    print("\nScenario Summary")

    scenario_names = sorted(set(r["scenario"] for r in results))

    for scenario in scenario_names:
        scenario_results = [
            r for r in results
            if r["scenario"] == scenario
        ]

        scenario_total = len(scenario_results)
        scenario_access_correct = sum(
            1 for r in scenario_results if r["access_correct"]
        )

        scenario_accuracy = (scenario_access_correct / scenario_total) * 100

        print(f"\n{scenario}")
        print(f"Attempts: {scenario_total}")
        print(f"Correct Access Decisions: {scenario_access_correct}")
        print(f"Access Accuracy: {scenario_accuracy:.2f}%")

    print("\n======================================================")


def run_test():
    results = []

    print("==============================================")
    print("Camera Recognition Accuracy + Anti-Spoof Test")
    print("==============================================")

    total_attempts = int(input("How many attempts do you want to record? "))

    for i in range(total_attempts):
        print(f"\nAttempt {i + 1}/{total_attempts}")

        scenario = choose_scenario()

        camera_result = open_camera_capture_and_test_antispoof(
            scenario,
            i + 1
        )

        if camera_result["actual_access"] == "quit":
            print("Attempt skipped.")
            continue

        expected_access = scenario["expected_access"]
        actual_access = camera_result["actual_access"]
        access_correct = expected_access == actual_access

        expected_spoof = scenario["expected_spoof"]
        actual_spoof = camera_result["spoof_result"]

        spoof_correct = False

        if actual_spoof in ["real", "fake"]:
            spoof_correct = expected_spoof == actual_spoof

        result = {
            "attempt": i + 1,
            "scenario": scenario["name"],
            "expected_access": expected_access,
            "actual_access": actual_access,
            "access_correct": access_correct,
            "expected_spoof": expected_spoof,
            "actual_spoof": actual_spoof,
            "spoof_correct": spoof_correct,
            "real_score": camera_result["real_score"],
            "fake_score": camera_result["fake_score"],
            "captured_image_path": camera_result["captured_image_path"],
            "anti_spoof_error": camera_result["anti_spoof_error"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        results.append(result)

        if access_correct:
            print("Access decision: Correct")
        else:
            print("Access decision: Wrong")

        if actual_spoof in ["real", "fake"]:
            if spoof_correct:
                print("Anti-spoof result: Correct")
            else:
                print("Anti-spoof result: Wrong")
        else:
            print("Anti-spoof result: Not recorded")

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    run_test()
