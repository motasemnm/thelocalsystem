import os
from firebase_client import get_firestore, get_bucket
from app.db import insert_home, insert_user, insert_membership, get_connection
from app.config import FACES_DIR


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ts_to_str(value):
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def insert_face_image(home_id, user_uid, image_name, local_path, cloud_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO face_images
        (home_id, user_uid, image_name, local_path, cloud_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    """, (
        home_id,
        user_uid,
        image_name,
        local_path,
        cloud_path
    ))

    conn.commit()
    conn.close()


def sync_homes():
    firestore_db = get_firestore()
    homes = firestore_db.collection("homes").stream()

    print("=== Syncing homes ===")
    for doc in homes:
        data = doc.to_dict()
        home_id = doc.id

        insert_home(
            home_id=home_id,
            name=data.get("homeName", ""),
            owner_uid=data.get("ownerUid", ""),
            created_at=ts_to_str(data.get("createdAt")),
            last_synced_at=None
        )

        print(f"✔ Synced home: {home_id} | {data.get('homeName')}")


def sync_users():
    firestore_db = get_firestore()
    users = firestore_db.collection("users").stream()

    print("=== Syncing users ===")
    for doc in users:
        data = doc.to_dict()
        user_uid = doc.id

        insert_user(
            uid=user_uid,
            name=data.get("name", ""),
            email=data.get("email", ""),
            phone=data.get("phone"),
            status=data.get("status", "active"),
            created_at=ts_to_str(data.get("createdAt")),
            updated_at=None
        )

        print(f"✔ Synced user: {user_uid} | {data.get('name')}")


def sync_memberships():
    firestore_db = get_firestore()
    memberships = firestore_db.collection("memberships").stream()

    print("=== Syncing memberships ===")
    for doc in memberships:
        data = doc.to_dict()

        home_id = data.get("homeRefId")
        user_uid = data.get("userRefId")

        if not home_id or not user_uid:
            print(f"⚠ Skipped invalid membership: {doc.id}")
            continue

        insert_membership(
            home_id=home_id,
            user_uid=user_uid,
            role=data.get("role", "member"),
            access=1 if data.get("access", True) else 0,
            status=data.get("status", "accepted"),
            created_at=ts_to_str(data.get("createdAt"))
        )

        print(f"✔ Synced membership: home={home_id}, user={user_uid}, role={data.get('role')}")


def sync_face_images():
    firestore_db = get_firestore()
    bucket = get_bucket()

    print("=== Downloading face images ===")

    memberships = firestore_db.collection("memberships").stream()

    for membership_doc in memberships:
        m_data = membership_doc.to_dict()
        home_id = m_data.get("homeRefId")
        user_uid = m_data.get("userRefId")

        if not home_id or not user_uid:
            continue

        # Use imagePath from membership if available,
        # otherwise build from standard new path structure
        image_path = m_data.get("imagePath")
        device_id  = m_data.get("deviceRefId")

        if image_path:
            prefix = image_path.rstrip("/") + "/"
        elif device_id:
            prefix = f"homes/{home_id}/member/{user_uid}/device/{device_id}/image/"
        else:
            prefix = f"homes/{home_id}/member/{user_uid}/"
        blobs = bucket.list_blobs(prefix=prefix)

        local_dir = os.path.join(FACES_DIR, f"home_{home_id}", f"user_{user_uid}")
        ensure_dir(local_dir)

        found_any = False

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            found_any = True
            file_name = os.path.basename(blob.name)
            local_path = os.path.join(local_dir, file_name)

            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"✔ Downloaded: {local_path}")
            else:
                print(f"ℹ Already exists: {local_path}")

            insert_face_image(
                home_id=home_id,
                user_uid=user_uid,
                image_name=file_name,
                local_path=local_path,
                cloud_path=blob.name
            )

        if not found_any:
            print(f"ℹ No face images found for home={home_id}, user={user_uid}")


def sync_all():
    sync_homes()
    sync_users()
    sync_memberships()
    sync_face_images()
    print("✅ Sync completed successfully")


if __name__ == "__main__":
    sync_all()
