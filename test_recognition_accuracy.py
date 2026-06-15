import cv2
import csv
from datetime import datetime


OUTPUT_FILE = "recognition_accuracy_camera_results.csv"
CAMERA_INDEX = 0


SCENARIOS = {
    "1": {
        "name": "Registered user with blink",
        "expected": "granted"
    },
    "2": {
        "name": "Registered user without blink",
        "expected": "denied"
    },
    "3": {
        "name": "Unknown user",
        "expected": "denied"
    },
    "4": {
        "name": "Photo / spoof attempt",
        "expected": "denied"
    }
}


def choose_scenario():
    print("\nChoose test scenario:")
    for key, value in SCENARIOS.items():
        print(f"{key}. {value['name']} | Expected: {value['expected']}")

    while True:
        choice = input("Enter scenario number: ").strip()

        if choice in SCENARIOS:
            return SCENARIOS[choice]

        print("Invalid choice. Choose 1, 2, 3, or 4.")


def open_camera_and_record_result(scenario):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise Exception("Camera could not be opened")

    actual_result = None

    print("\nCamera opened.")
    print("Perform the test in front of the camera.")
    print("Press G if the system result is GRANTED.")
    print("Press D if the system result is DENIED.")
    print("Press Q to quit this attempt.")

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            raise Exception("Could not read frame from camera")

        cv2.putText(
            frame,
            f"Scenario: {scenario['name']}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Expected: {scenario['expected']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press G = Granted | D = Denied | Q = Quit",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Recognition Accuracy Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("g"):
            actual_result = "granted"
            break
        elif key == ord("d"):
            actual_result = "denied"
            break
        elif key == ord("q"):
            actual_result = "quit"
            break

    cap.release()
    cv2.destroyAllWindows()

    return actual_result


def save_results(results):
    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "attempt",
                "scenario",
                "expected_result",
                "actual_result",
                "correct",
                "timestamp"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_FILE}")


def calculate_accuracy(results):
    if not results:
        return 0

    correct = sum(1 for r in results if r["correct"])
    return (correct / len(results)) * 100


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
        if r["actual_result"] == "granted"
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
        if r["actual_result"] == "denied"
    ]

    if not valid_attempts:
        return 0, 0, 0

    frr = (len(false_rejections) / len(valid_attempts)) * 100
    return frr, len(false_rejections), len(valid_attempts)


def print_summary(results):
    print("\n========== Recognition Accuracy Summary ==========")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    wrong = total - correct

    accuracy = calculate_accuracy(results)
    far, false_accepts, attack_attempts = calculate_false_acceptance_rate(results)
    frr, false_rejections, valid_attempts = calculate_false_rejection_rate(results)

    print(f"Total Attempts: {total}")
    print(f"Correct Decisions: {correct}")
    print(f"Wrong Decisions: {wrong}")
    print(f"Accuracy: {accuracy:.2f}%")

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
        scenario_correct = sum(1 for r in scenario_results if r["correct"])
        scenario_accuracy = (scenario_correct / scenario_total) * 100

        print(f"\n{scenario}")
        print(f"Attempts: {scenario_total}")
        print(f"Correct: {scenario_correct}")
        print(f"Accuracy: {scenario_accuracy:.2f}%")

    print("\n=================================================")


def run_test():
    results = []

    print("====================================")
    print("Camera-Based Recognition Accuracy Test")
    print("====================================")

    total_attempts = int(input("How many attempts do you want to record? "))

    for i in range(total_attempts):
        print(f"\nAttempt {i + 1}/{total_attempts}")

        scenario = choose_scenario()
        actual_result = open_camera_and_record_result(scenario)

        if actual_result == "quit":
            print("Attempt skipped.")
            continue

        expected = scenario["expected"]
        correct = expected == actual_result

        result = {
            "attempt": i + 1,
            "scenario": scenario["name"],
            "expected_result": expected,
            "actual_result": actual_result,
            "correct": correct,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        results.append(result)

        if correct:
            print("Result: Correct")
        else:
            print("Result: Wrong")

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    run_test()
