import hashlib
import sys
from datetime import datetime, timezone

from app.db import (
    get_connection,
    get_last_batch_hash,
    insert_blockchain_batch,
    update_batch_status,
    insert_blockchain_audit,
    get_unanchored_logs,
    mark_logs_batched,
)

def _recompute_log_hash(log: dict) -> str:
    raw = (
        f"{log['home_id']}|{log['device_id']}|{log['user_uid']}|"
        f"{log['result']}|{log['action']}|{log['confidence']}|"
        f"{log['spoof_result']}|{log['log_time']}|{log['previous_log_hash']}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_batch_hash(log_hashes: list, previous_batch_hash: str) -> str:
    combined = "|".join(log_hashes) + "|" + str(previous_batch_hash)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _get_all_logs_for_home(home_id: str) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, home_id, device_id, user_uid, result, action,
               confidence, spoof_result, log_time,
               log_hash, previous_log_hash, batch_id, blockchain_status
        FROM access_logs
        WHERE home_id = ?
        ORDER BY id ASC
    """, (home_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def _get_firebase_anchor(batch_id: str) -> dict | None:
    try:
        from app.firebase_client import get_firestore
        db = get_firestore()
        doc = db.collection("blockchain_anchors").document(batch_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"  ⚠ Could not reach Firebase for anchor check: {e}")
        return None



def verify_chain(home_id: str) -> dict:
    logs = _get_all_logs_for_home(home_id)

    if not logs:
        return {
            "valid": True,
            "total": 0,
            "checked": 0,
            "broken_at_log_id": None,
            "firebase_anchor_valid": None,
            "reason": "No logs found for this home.",
        }

    print(f"Verifying local chain for home: {home_id} ({len(logs)} logs)")

    previous_hash = None

    for i, log in enumerate(logs):
        log_id = log["id"]

        if log["previous_log_hash"] != previous_hash:
            reason = (
                f"Log {log_id}: previous_log_hash mismatch. "
                f"Expected '{previous_hash}', "
                f"found '{log['previous_log_hash']}'"
            )
            print(f"BROKEN at log {log_id}: previous_log_hash mismatch")
            return {
                "valid": False,
                "total": len(logs),
                "checked": i + 1,
                "broken_at_log_id": log_id,
                "firebase_anchor_valid": None,
                "reason": reason,
            }

        expected_hash = _recompute_log_hash(log)
        if log["log_hash"] != expected_hash:
            reason = (
                f"Log {log_id}: log_hash mismatch — "
                f"the log entry may have been tampered with."
            )
            print(f"BROKEN at log {log_id}: log_hash mismatch (tampered data)")
            return {
                "valid": False,
                "total": len(logs),
                "checked": i + 1,
                "broken_at_log_id": log_id,
                "firebase_anchor_valid": None,
                "reason": reason,
            }

        previous_hash = log["log_hash"]
        print(f"  ✔ Log {log_id} OK")

    print(f"Local chain intact. All {len(logs)} logs verified.")

    # ── Step 2: verify Firebase anchors ──
    # Fetch all batches for this home and compare hashes against Firebase
    print(f"\nVerifying Firebase anchors...")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, batch_hash, blockchain_status
        FROM blockchain_batches
        WHERE home_id = ? AND blockchain_status = 'anchored'
        ORDER BY created_at ASC
    """, (home_id,))
    batches = [dict(r) for r in cursor.fetchall()]
    conn.close()

    if not batches:
        print("  ℹ No anchored batches found — skipping Firebase check")
        return {
            "valid": True,
            "total": len(logs),
            "checked": len(logs),
            "broken_at_log_id": None,
            "firebase_anchor_valid": None,
            "reason": None,
        }

    firebase_valid = True
    for batch in batches:
        batch_id   = batch["id"]
        local_hash = batch["batch_hash"]

        anchor = _get_firebase_anchor(batch_id)

        if anchor is None:
            print(f"  ⚠ Batch {batch_id}: no Firebase anchor found")
            firebase_valid = False
            continue

        firebase_hash = anchor.get("batch_hash")
        if firebase_hash != local_hash:
            print(
                f"     Batch {batch_id}: Firebase hash mismatch!\n"
                f"     Local   : {local_hash}\n"
                f"     Firebase: {firebase_hash}\n"
                f"     Local logs were tampered with after anchoring!"
            )
            firebase_valid = False
        else:
            print(f"   Batch {batch_id}: Firebase anchor matches")

    if firebase_valid:
        print("All Firebase anchors match — chain is fully verified")
    else:
        print("Firebase anchor mismatch — TAMPERING DETECTED")

    return {
        "valid": True,
        "total": len(logs),
        "checked": len(logs),
        "broken_at_log_id": None,
        "firebase_anchor_valid": firebase_valid,
        "reason": None if firebase_valid else "Firebase anchor mismatch — tampering detected",
    }

def create_batch(home_id: str) -> dict | None:
    logs = get_unanchored_logs(home_id)

    if not logs:
        print("ℹ No unanchored logs to batch.")
        return None

    log_ids    = [log["id"] for log in logs]
    log_hashes = [log["log_hash"] for log in logs]

    previous_batch_hash = get_last_batch_hash(home_id)
    batch_hash = _compute_batch_hash(log_hashes, previous_batch_hash)

    print(f"   Creating batch for home {home_id}")
    print(f"   Logs included : {log_ids}")
    print(f"   Batch hash    : {batch_hash}")
    print(f"   Previous batch: {previous_batch_hash}")

    batch_id = insert_blockchain_batch(
        home_id=home_id,
        batch_hash=batch_hash,
        previous_batch_hash=previous_batch_hash,
        log_count=len(log_ids),
        blockchain_network="firebase",
        blockchain_tx_id=None,
        blockchain_status="pending",
        anchored_at=None,
    )

    mark_logs_batched(log_ids, batch_id)

    insert_blockchain_audit(
        batch_id=batch_id,
        action="batch_created",
        status="success",
        message=f"Batch created with {len(log_ids)} logs.",
    )

    print(f"Batch created: {batch_id}")

    return {
        "batch_id":            batch_id,
        "home_id":             home_id,
        "batch_hash":          batch_hash,
        "previous_batch_hash": previous_batch_hash,
        "log_count":           len(log_ids),
        "log_ids":             log_ids,
    }


def anchor_batch(batch_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blockchain_batches WHERE id = ?", (batch_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Batch not found: {batch_id}")
        return {"success": False, "reason": "Batch not found"}

    batch = dict(row)

    if batch["blockchain_status"] == "anchored":
        print(f"ℹ Batch {batch_id} is already anchored.")
        return {"success": False, "reason": "Already anchored"}

    print(f"⛓  Anchoring batch {batch_id} to Firebase...")

    try:
        from app.firebase_client import get_firestore
        db = get_firestore()

        now = datetime.now(timezone.utc).isoformat()

        anchor_data = {
            "batch_id":            batch_id,
            "home_id":             batch["home_id"],
            "batch_hash":          batch["batch_hash"],
            "previous_batch_hash": batch["previous_batch_hash"],
            "log_count":           batch["log_count"],
            "anchored_at":         now,
            "network":             "firebase",
        }

        # Write to Firebase — this is the external tamper-evident record
        db.collection("blockchain_anchors") \
          .document(batch_id) \
          .set(anchor_data)

        # Update local DB
        update_batch_status(
            batch_id=batch_id,
            blockchain_status="anchored",
            blockchain_tx_id=f"firebase:{batch_id}",
            anchored_at=now,
        )

        insert_blockchain_audit(
            batch_id=batch_id,
            action="batch_anchored",
            status="success",
            message=f"Anchored to Firebase Firestore. batch_id={batch_id}",
        )

        print(f" Batch anchored to Firebase: blockchain_anchors/{batch_id}")
        return {
            "success":     True,
            "batch_id":    batch_id,
            "batch_hash":  batch["batch_hash"],
            "anchored_at": now,
            "network":     "firebase",
        }

    except Exception as e:
        insert_blockchain_audit(
            batch_id=batch_id,
            action="batch_anchor_failed",
            status="error",
            message=str(e),
        )
        print(f" Anchoring failed: {e}")
        return {"success": False, "reason": str(e)}



def get_status(home_id: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN batch_id IS NULL THEN 1 ELSE 0 END) as unanchored,
               SUM(CASE WHEN blockchain_status = 'anchored' THEN 1 ELSE 0 END) as anchored
        FROM access_logs WHERE home_id = ?
    """, (home_id,))
    log_stats = dict(cursor.fetchone())

    cursor.execute("""
        SELECT COUNT(*) as total_batches,
               SUM(CASE WHEN blockchain_status = 'anchored' THEN 1 ELSE 0 END) as anchored_batches
        FROM blockchain_batches WHERE home_id = ?
    """, (home_id,))
    batch_stats = dict(cursor.fetchone())
    conn.close()

    print(f"\n Blockchain status for home: {home_id}")
    print(f"   Total logs      : {log_stats['total']}")
    print(f"   Unanchored logs : {log_stats['unanchored']}")
    print(f"   Anchored logs   : {log_stats['anchored']}")
    print(f"   Total batches   : {batch_stats['total_batches']}")
    print(f"   Anchored batches: {batch_stats['anchored_batches']}")

    return {**log_stats, **batch_stats}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m app.blockchain verify <home_id>")
        print("  python -m app.blockchain batch  <home_id>")
        print("  python -m app.blockchain anchor <batch_id>")
        print("  python -m app.blockchain status <home_id>")
        sys.exit(1)

    command = sys.argv[1].lower()
    arg     = sys.argv[2]

    if command == "verify":
        result = verify_chain(arg)
        print("\nResult:", result)

    elif command == "batch":
        result = create_batch(arg)
        print("\nResult:", result)

    elif command == "anchor":
        result = anchor_batch(arg)
        print("\nResult:", result)

    elif command == "status":
        get_status(arg)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
