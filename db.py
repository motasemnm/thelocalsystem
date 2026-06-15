import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone

from app.config import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _safe(value):
    return "" if value is None else str(value)


def test_db_connection():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table["name"] for table in tables]


def insert_home(home_id, name, owner_uid, created_at=None, last_synced_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO homes (id, name, owner_uid, created_at, last_synced_at)
        VALUES (?, ?, ?, ?, ?)
    """, (home_id, name, owner_uid, created_at, last_synced_at))

    conn.commit()
    conn.close()


def insert_user(uid, name, email, phone=None, status="active",
                created_at=None, updated_at=None):
    """
    Safe user insert/update.

    This avoids:
    UNIQUE constraint failed: users.email

    Reason:
    Sometimes the same email already exists locally from an old home/cache.
    Instead of crashing, we update the existing user record.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT uid FROM users WHERE uid = ?", (uid,))
        existing_by_uid = cursor.fetchone()

        if existing_by_uid:
            cursor.execute("""
                UPDATE users
                SET name = ?, email = ?, phone = ?, status = ?,
                    created_at = ?, updated_at = ?
                WHERE uid = ?
            """, (name, email, phone, status, created_at, updated_at, uid))

        else:
            cursor.execute("SELECT uid FROM users WHERE email = ?", (email,))
            existing_by_email = cursor.fetchone()

            if existing_by_email:
                old_uid = existing_by_email["uid"]

                cursor.execute("""
                    UPDATE users
                    SET uid = ?, name = ?, email = ?, phone = ?, status = ?,
                        created_at = ?, updated_at = ?
                    WHERE uid = ?
                """, (uid, name, email, phone, status,
                      created_at, updated_at, old_uid))

            else:
                cursor.execute("""
                    INSERT INTO users (
                        uid, name, email, phone, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (uid, name, email, phone, status,
                      created_at, updated_at))

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"insert_user skipped/error: {e}")

    finally:
        conn.close()



def insert_membership(home_id, user_uid, role="member", access=1,
                      status="accepted", created_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id FROM memberships
        WHERE home_id = ? AND user_uid = ?
    """, (home_id, user_uid))

    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE memberships
            SET role = ?, access = ?, status = ?, created_at = ?
            WHERE id = ?
        """, (role, access, status, created_at, existing["id"]))
    else:
        cursor.execute("""
            INSERT INTO memberships (
                home_id, user_uid, role, access, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (home_id, user_uid, role, access, status, created_at))

    conn.commit()
    conn.close()



def insert_device(device_id, home_id, device_name, device_type=None,
                  device_status="offline", last_seen=None, created_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO devices (
            id, home_id, device_name, device_type,
            device_status, last_seen, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        device_id, home_id, device_name, device_type,
        device_status, last_seen, created_at
    ))

    conn.commit()
    conn.close()


def update_device_status(device_id, status, last_seen=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE devices
        SET device_status = ?, last_seen = ?
        WHERE id = ?
    """, (status, last_seen, device_id))

    conn.commit()
    conn.close()



def insert_face_image(home_id, user_uid, image_name, local_path,
                      cloud_path=None, download_url=None, embedding_path=None,
                      status="active", created_at=None, updated_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, embedding_path FROM face_images
        WHERE home_id = ? AND user_uid = ? AND image_name = ?
    """, (home_id, user_uid, image_name))

    existing = cursor.fetchone()

    if existing:
        final_embedding = (
            embedding_path
            if embedding_path is not None
            else existing["embedding_path"]
        )

        cursor.execute("""
            UPDATE face_images
            SET local_path = ?, cloud_path = ?, download_url = ?,
                embedding_path = ?, status = ?, updated_at = ?
            WHERE id = ?
        """, (
            local_path, cloud_path, download_url,
            final_embedding, status, updated_at, existing["id"]
        ))

    else:
        cursor.execute("""
            INSERT INTO face_images (
                home_id, user_uid, image_name, local_path,
                cloud_path, download_url, embedding_path,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            home_id, user_uid, image_name, local_path,
            cloud_path, download_url, embedding_path,
            status, created_at, updated_at
        ))

    conn.commit()
    conn.close()

def insert_model(model_name, version, local_path, cloud_path=None,
                 active=1, downloaded_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id FROM models
        WHERE model_name = ? AND version = ?
    """, (model_name, version))

    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE models
            SET local_path = ?, cloud_path = ?, active = ?, downloaded_at = ?
            WHERE id = ?
        """, (
            local_path, cloud_path, active,
            downloaded_at, existing["id"]
        ))

    else:
        cursor.execute("""
            INSERT INTO models (
                model_name, version, local_path,
                cloud_path, active, downloaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_name, version, local_path,
            cloud_path, active, downloaded_at
        ))

    conn.commit()
    conn.close()


def get_active_model(model_name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM models
        WHERE model_name = ? AND active = 1
        ORDER BY id DESC
        LIMIT 1
    """, (model_name,))

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def get_last_log_hash(home_id, device_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT log_hash FROM access_logs
        WHERE home_id = ? AND device_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (home_id, device_id))

    row = cursor.fetchone()
    conn.close()

    return row["log_hash"] if row else None


def get_previous_log_hash(home_id, device_id):
    """
    Get previous hash from local SQLite only.

    Important:
    This version does NOT call deviceGetLastLogHash from Firebase,
    because that Cloud Function does not exist in your project.

    If no local log exists, this starts a new local hash chain.
    """
    previous_log_hash = get_last_log_hash(home_id, device_id)

    if previous_log_hash is not None:
        return previous_log_hash

    print("No local previous log hash found. Starting new local hash chain.")
    return None


def build_log_hash(home_id, device_id, user_uid, result, action,
                   confidence, spoof_result, log_time, previous_log_hash):
    raw = (
        f"{_safe(home_id)}|"
        f"{_safe(device_id)}|"
        f"{_safe(user_uid)}|"
        f"{_safe(result)}|"
        f"{_safe(action)}|"
        f"{_safe(confidence)}|"
        f"{_safe(spoof_result)}|"
        f"{_safe(log_time)}|"
        f"{_safe(previous_log_hash)}"
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def insert_access_log(home_id, device_id, user_uid, result, action,
                      confidence=None, spoof_result=None,
                      face_image_id=None,
                      uploaded_to_cloud=0, blockchain_status="pending"):
    log_time = datetime.now(timezone.utc).isoformat()

    previous_log_hash = get_previous_log_hash(home_id, device_id)

    log_hash = build_log_hash(
        home_id=home_id,
        device_id=device_id,
        user_uid=user_uid,
        result=result,
        action=action,
        confidence=confidence,
        spoof_result=spoof_result,
        log_time=log_time,
        previous_log_hash=previous_log_hash,
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO access_logs (
            home_id, device_id, user_uid, face_image_id,
            result, action, confidence, spoof_result,
            log_time, uploaded_to_cloud,
            log_hash, previous_log_hash,
            batch_id, blockchain_status,
            blockchain_tx_id, blockchain_anchor_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
    """, (
        home_id, device_id, user_uid, face_image_id,
        result, action, confidence, spoof_result,
        log_time, uploaded_to_cloud,
        log_hash, previous_log_hash,
        blockchain_status,
    ))

    log_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        "id": log_id,
        "home_id": home_id,
        "device_id": device_id,
        "user_uid": user_uid,
        "face_image_id": face_image_id,
        "result": result,
        "action": action,
        "confidence": confidence,
        "spoof_result": spoof_result,
        "log_time": log_time,
        "uploaded_to_cloud": uploaded_to_cloud,
        "log_hash": log_hash,
        "previous_log_hash": previous_log_hash,
        "blockchain_status": blockchain_status,
    }


def get_unanchored_logs(home_id, limit=50):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM access_logs
        WHERE home_id = ? AND batch_id IS NULL
        ORDER BY id ASC
        LIMIT ?
    """, (home_id, limit))

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows


def mark_logs_batched(log_ids, batch_id):
    if not log_ids:
        return

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(log_ids))

    cursor.execute(f"""
        UPDATE access_logs
        SET batch_id = ?, blockchain_status = 'anchored'
        WHERE id IN ({placeholders})
    """, [batch_id] + list(log_ids))

    conn.commit()
    conn.close()



def get_last_batch_hash(home_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT batch_hash FROM blockchain_batches
        WHERE home_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (home_id,))

    row = cursor.fetchone()
    conn.close()

    return row["batch_hash"] if row else None


def insert_blockchain_batch(home_id, batch_hash, previous_batch_hash,
                            log_count, blockchain_network=None,
                            blockchain_tx_id=None,
                            blockchain_status="pending",
                            anchored_at=None):
    batch_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO blockchain_batches (
            id, home_id, batch_hash, previous_batch_hash,
            log_count, blockchain_network, blockchain_tx_id,
            blockchain_status, anchored_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id, home_id, batch_hash, previous_batch_hash,
        log_count, blockchain_network, blockchain_tx_id,
        blockchain_status, anchored_at, created_at
    ))

    conn.commit()
    conn.close()

    return batch_id


def update_batch_status(batch_id, blockchain_status,
                        blockchain_tx_id=None, anchored_at=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE blockchain_batches
        SET blockchain_status = ?, blockchain_tx_id = ?, anchored_at = ?
        WHERE id = ?
    """, (
        blockchain_status, blockchain_tx_id,
        anchored_at, batch_id
    ))

    conn.commit()
    conn.close()


def insert_blockchain_audit(batch_id, action, status, message=None):
    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO blockchain_audit (
            batch_id, action, status, message, created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        batch_id, action, status,
        message, created_at
    ))

    conn.commit()
    conn.close()




def enqueue_sync(entity_type, entity_id, operation):
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sync_queue (
            entity_type, entity_id, operation,
            status, created_at, updated_at
        )
        VALUES (?, ?, ?, 'pending', ?, ?)
    """, (
        entity_type, entity_id,
        operation, now, now
    ))

    conn.commit()
    conn.close()


def get_pending_sync_items():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM sync_queue
        WHERE status = 'pending'
        ORDER BY id ASC
    """)

    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    return rows


def update_sync_status(item_id, status, last_error=None):
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE sync_queue
        SET status = ?, last_error = ?, updated_at = ?
        WHERE id = ?
    """, (
        status, last_error,
        now, item_id
    ))

    conn.commit()
    conn.close()
