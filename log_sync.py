from app.device_api import upload_log
from app.db import get_connection, enqueue_sync


def upload_pending_logs():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, home_id, device_id, user_uid, result, action,
               confidence, spoof_result, log_time,
               uploaded_to_cloud, log_hash, previous_log_hash,
               blockchain_status
        FROM access_logs
        WHERE uploaded_to_cloud = 0
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()

    if not rows:
        print("ℹ No pending logs to upload")
        conn.close()
        return

    print("=== Uploading pending logs via API ===")

    for row in rows:
        log_id = str(row["id"])

        log_data = {
            "homeId": row["home_id"],
            "deviceId": row["device_id"],
            "userUid": row["user_uid"],
            "result": row["result"],
            "action": row["action"],
            "confidence": row["confidence"],
            "spoofResult": row["spoof_result"],
            "logTime": row["log_time"],
            "logHash": row["log_hash"],
            "previousLogHash": row["previous_log_hash"],
            "blockchainStatus": row["blockchain_status"],
        }

        try:
            upload_log(log_id, log_data)

            cursor.execute("""
                UPDATE access_logs
                SET uploaded_to_cloud = 1
                WHERE id = ?
            """, (row["id"],))

            conn.commit()
            print(f" Uploaded log {log_id} via API")

        except Exception as e:
            print(f" Failed to upload log {log_id}: {e}")

            try:
                enqueue_sync("log", log_id, "upload")
                print(f" Queued log {log_id} for retry")
            except Exception as queue_error:
                print(f" Could not queue log {log_id}: {queue_error}")

    conn.close()
    print("Log upload process complete")


if __name__ == "__main__":
    upload_pending_logs()
