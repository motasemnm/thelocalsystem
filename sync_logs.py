"""
sync_logs.py
------------
Offline sync queue handler.

Processes the sync_queue table — retries any operations
that failed or were queued while the internet was offline.

Currently handles:
  - 'log' entities    → re-uploads access logs to Firebase
  - 'device_status'   → re-syncs device online/offline status

Usage:
    python -m app.sync_logs
"""

from app.firebase_client import get_firestore
from app.db import (
    get_connection,
    get_pending_sync_items,
    update_sync_status,
)


def process_sync_queue():
    firestore_db = get_firestore()
    items = get_pending_sync_items()

    if not items:
        print("ℹ No pending items in sync queue")
        return

    print(f"=== Processing sync queue ({len(items)} items) ===")

    for item in items:
        item_id     = item["id"]
        entity_type = item["entity_type"]
        entity_id   = item["entity_id"]
        operation   = item["operation"]

        try:
            if entity_type == "log":
                _sync_log(firestore_db, entity_id)

            elif entity_type == "device_status":
                _sync_device_status(firestore_db, entity_id)

            else:
                print(f"⚠ Unknown entity type: {entity_type} — skipping")
                update_sync_status(item_id, "skipped")
                continue

            update_sync_status(item_id, "done")
            print(f"✔ Synced {entity_type} {entity_id}")

        except Exception as e:
            update_sync_status(item_id, "failed", last_error=str(e))
            print(f"❌ Failed to sync {entity_type} {entity_id}: {e}")

    print("✅ Sync queue processing complete")


def _sync_log(firestore_db, log_id):
    """Re-upload a single access log to Firebase."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise Exception(f"Log {log_id} not found in local DB")

    log_data = {
        "homeId":            row["home_id"],
        "deviceId":          row["device_id"],
        "userUid":           row["user_uid"],
        "result":            row["result"],
        "action":            row["action"],
        "confidence":        row["confidence"],
        "spoofResult":       row["spoof_result"],
        "logTime":           row["log_time"],
        "logHash":           row["log_hash"],
        "previousLogHash":   row["previous_log_hash"],
        "blockchainStatus":  row["blockchain_status"],
    }

    firestore_db \
        .collection("homes") \
        .document(row["home_id"]) \
        .collection("devices") \
        .document(row["device_id"]) \
        .collection("logs") \
        .document(str(log_id)) \
        .set(log_data)

    # Mark as uploaded in local DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE access_logs SET uploaded_to_cloud = 1 WHERE id = ?",
        (log_id,)
    )
    conn.commit()
    conn.close()


def _sync_device_status(firestore_db, device_id):
    """Re-sync a device's status to Firebase."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM devices WHERE id = ?", (device_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise Exception(f"Device {device_id} not found in local DB")

    from datetime import datetime, timezone
    firestore_db \
        .collection("homes") \
        .document(row["home_id"]) \
        .collection("devices") \
        .document(device_id) \
        .update({
            "deviceStatus": row["device_status"],
            "lastSeen":     datetime.now(timezone.utc),
        })


if __name__ == "__main__":
    process_sync_queue()
