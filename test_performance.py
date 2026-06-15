import time
import csv
import sqlite3
import statistics
import psutil
import cv2
import os
from datetime import datetime


# Change this path based on your real database location
# If your database is in the same folder as this script, use: "smart_home.db"
# If your real project path is /opt/smart_home/data/smart_home.db, use that path.
DB_PATH = "smart_home.db"

TEST_ROUNDS = 20
CAMERA_INDEX = 0
RESULTS_FILE = "performance_results.csv"


def measure_time(function, name):
    """
    Measures execution time, success/failure, CPU usage, RAM usage,
    and timestamp for any test function.
    """
    start = time.perf_counter()
    success = True
    error = None
    extra_info = None

    try:
        result = function()
        if result is not None:
            extra_info = result
    except Exception as e:
        success = False
        error = str(e)

    end = time.perf_counter()

    return {
        "test_name": name,
        "time_seconds": round(end - start, 4),
        "success": success,
        "error": error,
        "extra_info": extra_info,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def test_camera_capture():
    """
    Tests if the camera can open and capture one frame.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise Exception("Camera could not be opened")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Frame could not be captured")

    return "Frame captured successfully"


def test_camera_fps():
    """
    Measures average camera FPS for 3 seconds.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise Exception("Camera could not be opened")

    frame_count = 0
    start = time.perf_counter()

    while time.perf_counter() - start < 3:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    cap.release()

    fps = frame_count / 3
    print(f"Average Camera FPS: {fps:.2f}")

    return f"Average FPS: {fps:.2f}"


def test_sqlite_insert():
    """
    Tests SQLite write performance by inserting one row.
    """
    ensure_database_folder_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_name TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        INSERT INTO performance_test (test_name, created_at)
        VALUES (?, ?)
    """, ("sqlite_insert_test", datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return "SQLite insert successful"


def test_sqlite_read():
    """
    Tests SQLite read performance.
    """
    ensure_database_folder_exists()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sqlite_master")
    count = cursor.fetchone()[0]

    conn.close()

    return f"SQLite read successful, tables count: {count}"


def test_full_local_decision_simulation():
    """
    This simulates the local access decision timing.

    Important:
    This is not the real FaceNet/blink/anti-spoofing measurement.
    It only estimates the general local processing pipeline.

    Later, you can replace the sleep values with real functions:
    - face detection
    - FaceNet recognition
    - blink detection
    - anti-spoofing
    - decision logic
    - SQLite logging
    """

    # Camera capture
    test_camera_capture()

    # Simulated face detection time
    time.sleep(0.08)

    # Simulated FaceNet embedding / comparison time
    time.sleep(0.25)

    # Simulated blink detection time
    time.sleep(0.04)

    # Simulated decision logic time
    time.sleep(0.01)

    # SQLite logging
    test_sqlite_insert()

    return "Full local decision simulation completed"


def ensure_database_folder_exists():
    """
    If DB_PATH includes folders, create them if they do not exist.
    Example: /opt/smart_home/data/smart_home.db
    """
    folder = os.path.dirname(DB_PATH)

    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def run_tests():
    results = []

    print("Starting local system performance test...\n")

    for i in range(TEST_ROUNDS):
        print(f"Round {i + 1}/{TEST_ROUNDS}")

        results.append(measure_time(test_camera_capture, "camera_capture"))
        results.append(measure_time(test_sqlite_insert, "sqlite_insert"))
        results.append(measure_time(test_sqlite_read, "sqlite_read"))
        results.append(measure_time(test_full_local_decision_simulation, "full_local_decision_simulation"))

    print("\nTesting camera FPS...")
    results.append(measure_time(test_camera_fps, "camera_fps"))

    save_results(results)
    print_summary(results)


def save_results(results):
    """
    Saves all test results into a CSV file.
    """
    with open(RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_name",
                "time_seconds",
                "success",
                "error",
                "extra_info",
                "cpu_percent",
                "ram_percent",
                "timestamp"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_FILE}")


def print_summary(results):
    """
    Prints average, minimum, maximum time, success count,
    failure count, CPU usage, and RAM usage.
    """
    print("\n========== Performance Summary ==========")

    test_names = sorted(set(r["test_name"] for r in results))

    for name in test_names:
        times = [
            r["time_seconds"]
            for r in results
            if r["test_name"] == name and r["success"]
        ]

        failures = [
            r for r in results
            if r["test_name"] == name and not r["success"]
        ]

        print(f"\n{name}")

        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)

            print(f"Average Time: {avg_time:.4f} seconds")
            print(f"Minimum Time: {min_time:.4f} seconds")
            print(f"Maximum Time: {max_time:.4f} seconds")
            print(f"Successful Tests: {len(times)}")
            print(f"Failed Tests: {len(failures)}")
        else:
            print("No successful tests")
            print(f"Failed Tests: {len(failures)}")

        if failures:
            print("Errors:")
            for failure in failures:
                print(f"- {failure['error']}")

    cpu_values = [r["cpu_percent"] for r in results]
    ram_values = [r["ram_percent"] for r in results]

    avg_cpu = statistics.mean(cpu_values)
    avg_ram = statistics.mean(ram_values)

    print("\nSystem Usage")
    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    print(f"Average RAM Usage: {avg_ram:.2f}%")

    print("\n=========================================")


if __name__ == "__main__":
    run_tests()
