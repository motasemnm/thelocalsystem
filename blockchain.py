import hashlib
import sys
from datetime import datetime, timezone

from app.device_api import call_function
from app.db import (
    get_connection,
    insert_blockchain_batch,
    update_batch_status,
    insert_blockchain_audit,
    mark_logs_batched,
)


def _safe(value):
    return "" if value is None else str(value)


def recompute_log_hash(log: dict) -> str:
    """
    Recalculate the hash of one access log.
    Must match the same fields used when the log was first created.
    """
    raw = (
        f"{_safe(log['home_id'])}|"
        f"{_safe(log['device_id'])}|"
        f"{_safe(log['user_uid'])}|"
        f"{_safe(log['result'])}|"
        f"{_safe(log['action'])}|"
        f"{_safe(log['confidence'])}|"
        f"{_safe(log['spoof_result'])}|"
        f"{_safe(log['log_time'])}|"
        f"{_safe(log['previous_log_hash'])}"
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_batch_hash(log_hashes: list, previous_batch_hash: str | None) -> str:
    combined = "|".join(log_hashes) + "|" + _safe(previous_batch_hash)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def get_all_logs_for_home_device(home_id: str, device_id: str) -> list:
    """
    Return logs only for the selected home and selected device.

    This makes blockchain validation independent for each home/device.
    Old logs from another home or device will not be checked here.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, home_id, device_id, user_uid, result, action,
               confidence, spoof_result, log_time,
               uploaded_to_cloud,
               log_hash, previous_log_hash,
               batch_id, blockchain_status,
               blockchain_tx_id, blockchain_anchor_time
        FROM access_logs
        WHERE home_id = ?
          AND device_id = ?
        ORDER BY id ASC
        """,
        (home_id, device_id),
    )

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_unanchored_logs_for_home_device(home_id: str, device_id: str, limit=50) -> list:
    """
    Return unbatched logs only for the selected home/device.
    This keeps blockchain batches independent per device.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM access_logs
        WHERE home_id = ?
          AND device_id = ?
          AND batch_id IS NULL
        ORDER BY id ASC
        LIMIT ?
        """,
        (home_id, device_id, limit),
    )

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_last_batch_hash_for_home_device(home_id: str, device_id: str) -> str | None:
    """
    Get the latest batch hash for this exact home/device.

    blockchain_batches does not store device_id directly, so we find batches
    through access_logs.batch_id for this home/device.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT b.batch_hash
        FROM blockchain_batches b
        WHERE b.id IN (
            SELECT DISTINCT batch_id
            FROM access_logs
            WHERE home_id = ?
              AND device_id = ?
              AND batch_id IS NOT NULL
        )
        ORDER BY b.created_at DESC
        LIMIT 1
        """,
        (home_id, device_id),
    )

    row = cursor.fetchone()
    conn.close()

    return row["batch_hash"] if row else None


def get_batches_for_home_device(home_id: str, device_id: str) -> list:
    """
    Return anchored batches connected to logs for this specific home/device.

    This avoids validating unrelated batches from another home/device.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT DISTINCT b.id, b.batch_hash, b.blockchain_status
        FROM blockchain_batches b
        INNER JOIN access_logs l ON l.batch_id = b.id
        WHERE l.home_id = ?
          AND l.device_id = ?
          AND b.blockchain_status = 'anchored'
        ORDER BY b.created_at ASC
        """,
        (home_id, device_id),
    )

    batches = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return batches


def get_firebase_anchor(batch_id: str) -> dict | None:
    try:
        result = call_function(
            "deviceGetBlockchainAnchor",
            {
                "batchId": batch_id,
            },
        )

        if not result or not result.get("exists"):
            return None

        return result.get("anchor")

    except Exception as e:
        print(f"⚠ Could not reach Firebase anchor API: {e}")
        return None


def verify_chain(home_id: str, device_id: str) -> dict:
    """
    Verify the local access log chain for one specific home/device.

    This prevents an old broken chain from another home/device from affecting
    the currently configured local system.
    """
    logs = get_all_logs_for_home_device(home_id, device_id)

    if not logs:
        print("No logs found for this home/device.")
        return {
            "valid": True,
            "home_id": home_id,
            "device_id": device_id,
            "total": 0,
            "checked": 0,
            "broken_at_log_id": None,
            "firebase_anchor_valid": None,
            "reason": "No logs found for this home/device.",
        }

    print(f"Verifying local chain for home: {home_id}")
    print(f"Verifying local chain for device: {device_id}")
    print(f"Total logs found: {len(logs)}")

    previous_hash = None

    for index, log in enumerate(logs):
        log_id = log["id"]

        stored_previous_hash = log.get("previous_log_hash")
        stored_log_hash = log.get("log_hash")

        if _safe(stored_previous_hash) != _safe(previous_hash):
            reason = (
                f"Log {log_id}: previous_log_hash mismatch. "
                f"Expected '{_safe(previous_hash)}', "
                f"found '{_safe(stored_previous_hash)}'."
            )

            print(f" BROKEN at log {log_id}: previous_log_hash mismatch")

            return {
                "valid": False,
                "home_id": home_id,
                "device_id": device_id,
                "total": len(logs),
                "checked": index + 1,
                "broken_at_log_id": log_id,
                "firebase_anchor_valid": None,
                "reason": reason,
            }

        expected_hash = recompute_log_hash(log)

        if _safe(stored_log_hash) != _safe(expected_hash):
            reason = (
                f"Log {log_id}: log_hash mismatch. "
                f"The log entry may have been changed after creation."
            )

            print(f" BROKEN at log {log_id}: log_hash mismatch")
            print(f"Stored  : {stored_log_hash}")
            print(f"Expected: {expected_hash}")

            return {
                "valid": False,
                "home_id": home_id,
                "device_id": device_id,
                "total": len(logs),
                "checked": index + 1,
                "broken_at_log_id": log_id,
                "firebase_anchor_valid": None,
                "reason": reason,
            }

        previous_hash = stored_log_hash
        print(f"✔ Log {log_id} OK")

    print(f"Local chain intact. All {len(logs)} logs verified.")

    print("Verifying Firebase anchors via API...")

    batches = get_batches_for_home_device(home_id, device_id)

    if not batches:
        print("ℹ No anchored batches found for this home/device — skipping Firebase anchor check.")
        return {
            "valid": True,
            "home_id": home_id,
            "device_id": device_id,
            "total": len(logs),
            "checked": len(logs),
            "broken_at_log_id": None,
            "firebase_anchor_valid": None,
            "reason": None,
        }

    firebase_valid = True

    for batch in batches:
        batch_id = batch["id"]
        local_hash = batch["batch_hash"]

        anchor = get_firebase_anchor(batch_id)

        if anchor is None:
            print(f" Batch {batch_id}: no Firebase anchor found.")
            firebase_valid = False
            continue

        firebase_hash = anchor.get("batch_hash")

        if firebase_hash != local_hash:
            print(f" Batch {batch_id}: Firebase anchor mismatch.")
            print(f"Local   : {local_hash}")
            print(f"Firebase: {firebase_hash}")
            firebase_valid = False
        else:
            print(f"✔ Batch {batch_id}: Firebase anchor matches.")

    return {
        "valid": True,
        "home_id": home_id,
        "device_id": device_id,
        "total": len(logs),
        "checked": len(logs),
        "broken_at_log_id": None,
        "firebase_anchor_valid": firebase_valid,
        "reason": None if firebase_valid else "Firebase anchor mismatch detected.",
    }


def create_batch(home_id: str, device_id: str) -> dict | None:
    """
    Create a blockchain batch only for unanchored logs belonging to the
    selected home/device.
    """
    logs = get_unanchored_logs_for_home_device(home_id, device_id)

    if not logs:
        print("ℹ No unanchored logs to batch for this home/device.")
        return None

    log_ids = [log["id"] for log in logs]
    log_hashes = [log["log_hash"] for log in logs]

    previous_batch_hash = get_last_batch_hash_for_home_device(home_id, device_id)
    batch_hash = compute_batch_hash(log_hashes, previous_batch_hash)

    print(f"Creating batch for home {home_id}")
    print(f"Creating batch for device {device_id}")
    print(f"Logs included : {log_ids}")
    print(f"Batch hash    : {batch_hash}")
    print(f"Previous batch: {previous_batch_hash}")

    batch_id = insert_blockchain_batch(
        home_id=home_id,
        batch_hash=batch_hash,
        previous_batch_hash=previous_batch_hash,
        log_count=len(log_ids),
        blockchain_network="firebase_cloud_function",
        blockchain_tx_id=None,
        blockchain_status="pending",
        anchored_at=None,
    )

    mark_logs_batched(log_ids, batch_id)

    insert_blockchain_audit(
        batch_id=batch_id,
        action="batch_created",
        status="success",
        message=f"Batch created with {len(log_ids)} logs for device {device_id}.",
    )

    print(f"Batch created: {batch_id}")

    return {
        "batch_id": batch_id,
        "home_id": home_id,
        "device_id": device_id,
        "batch_hash": batch_hash,
        "previous_batch_hash": previous_batch_hash,
        "log_count": len(log_ids),
        "log_ids": log_ids,
    }


def anchor_batch(batch_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM blockchain_batches WHERE id = ?",
        (batch_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Batch not found: {batch_id}")
        return {
            "success": False,
            "reason": "Batch not found",
        }

    batch = dict(row)

    if batch["blockchain_status"] == "anchored":
        print(f"ℹ Batch {batch_id} is already anchored.")
        return {
            "success": False,
            "reason": "Already anchored",
        }

    print(f"Anchoring batch {batch_id} via Cloud Function...")

    try:
        now = datetime.now(timezone.utc).isoformat()

        call_function(
            "deviceAnchorBatch",
            {
                "batchId": batch_id,
                "batchHash": batch["batch_hash"],
                "previousBatchHash": batch["previous_batch_hash"],
                "logCount": batch["log_count"],
            },
        )

        update_batch_status(
            batch_id=batch_id,
            blockchain_status="anchored",
            blockchain_tx_id=f"firebase_cloud_function:{batch_id}",
            anchored_at=now,
        )

        insert_blockchain_audit(
            batch_id=batch_id,
            action="batch_anchored",
            status="success",
            message=f"Anchored via Cloud Function. batch_id={batch_id}",
        )

        print(f"Batch anchored via Cloud Function: {batch_id}")

        return {
            "success": True,
            "batch_id": batch_id,
            "batch_hash": batch["batch_hash"],
            "anchored_at": now,
            "network": "firebase_cloud_function",
        }

    except Exception as e:
        insert_blockchain_audit(
            batch_id=batch_id,
            action="batch_anchor_failed",
            status="error",
            message=str(e),
        )

        print(f"Anchoring failed: {e}")

        return {
            "success": False,
            "reason": str(e),
        }


def get_status(home_id: str, device_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) as total,
               SUM(CASE WHEN batch_id IS NULL THEN 1 ELSE 0 END) as unanchored,
               SUM(CASE WHEN blockchain_status = 'anchored' THEN 1 ELSE 0 END) as anchored
        FROM access_logs
        WHERE home_id = ?
          AND device_id = ?
        """,
        (home_id, device_id),
    )

    log_stats = dict(cursor.fetchone())

    cursor.execute(
        """
        SELECT COUNT(DISTINCT b.id) as total_batches,
               SUM(CASE WHEN b.blockchain_status = 'anchored' THEN 1 ELSE 0 END) as anchored_batches
        FROM blockchain_batches b
        INNER JOIN access_logs l ON l.batch_id = b.id
        WHERE l.home_id = ?
          AND l.device_id = ?
        """,
        (home_id, device_id),
    )

    batch_stats = dict(cursor.fetchone())
    conn.close()

    print(f"Blockchain status for home: {home_id}")
    print(f"Blockchain status for device: {device_id}")
    print(f"Total logs      : {log_stats['total']}")
    print(f"Unanchored logs : {log_stats['unanchored']}")
    print(f"Anchored logs   : {log_stats['anchored']}")
    print(f"Total batches   : {batch_stats['total_batches']}")
    print(f"Anchored batches: {batch_stats['anchored_batches']}")

    return {
        **log_stats,
        **batch_stats,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 -m app.blockchain verify <home_id> <device_id>")
        print("  python3 -m app.blockchain batch  <home_id> <device_id>")
        print("  python3 -m app.blockchain anchor <batch_id>")
        print("  python3 -m app.blockchain status <home_id> <device_id>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "verify":
        if len(sys.argv) < 4:
            print("Usage: python3 -m app.blockchain verify <home_id> <device_id>")
            sys.exit(1)

        home_id = sys.argv[2]
        device_id = sys.argv[3]
        result = verify_chain(home_id, device_id)
        print("\nResult:", result)

    elif command == "batch":
        if len(sys.argv) < 4:
            print("Usage: python3 -m app.blockchain batch <home_id> <device_id>")
            sys.exit(1)

        home_id = sys.argv[2]
        device_id = sys.argv[3]
        result = create_batch(home_id, device_id)
        print("\nResult:", result)

    elif command == "anchor":
        batch_id = sys.argv[2]
        result = anchor_batch(batch_id)
        print("\nResult:", result)

    elif command == "status":
        if len(sys.argv) < 4:
            print("Usage: python3 -m app.blockchain status <home_id> <device_id>")
            sys.exit(1)

        home_id = sys.argv[2]
        device_id = sys.argv[3]
        result = get_status(home_id, device_id)
        print("\nResult:", result)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
