from app.device_api import upload_log, update_device_status
from app.db import (
    get_connection,
    get_pending_sync_items,
    update_sync_status,
)


def process_sync_queue():
    items = get_pending_sync_items()

    if not items:
        print("ℹ No pending items in sync queue")
        return

    print(f"=== Processing sync queue ({len(items)} items) ===")

    for item in items:
        item_id = item["id"]
        entity_type = item["entity_type"]
        entity_id = item["entity_id"]

        try:
            if entity_type == "log":
                _sync_log(entity_id)

            elif entity_type == "device_status":
                _sync_device_status(entity_id)

            else:
                print(f"⚠ Unknown entity type: {entity_type} — skipping")
                update_sync_status(item_id, "skipped")
                continue

            update_sync_status(item_id, "done")
            print(f" Synced {entity_type} {entity_id}")

        except Exception as e:
            update_sync_status(item_id, "failed", last_error=str(e))
            print(f" Failed to sync {entity_type} {entity_id}: {e}")

    print(" Sync queue processing complete")


def _sync_log(log_id):
    """Re-upload a single access log via API."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise Exception(f"Log {log_id} not found in local DB")

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

    upload_log(log_id, log_data)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE access_logs SET uploaded_to_cloud = 1 WHERE id = ?",
        (log_id,),
    )
    conn.commit()
    conn.close()


def _sync_device_status(device_id):
    """Re-sync device status via API."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise Exception(f"Device {device_id} not found in local DB")

    status = row["device_status"]

    update_device_status(status)


if __name__ == "__main__":
    process_sync_queue()
