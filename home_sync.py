import os
from datetime import datetime, timezone

from app.firebase_client import get_firestore, get_bucket
from google.cloud import firestore
from app.db import (
    insert_home,
    insert_user,
    insert_membership,
    insert_device,
    insert_face_image,
    get_connection,
)
from app.config import FACES_DIR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ts_to_str(value):
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sync_selected_home(home_id, device_id):
    firestore_db = get_firestore()
    bucket = get_bucket()

    # =========================
    # 1) Verify home
    # =========================
    home_ref = firestore_db.collection("homes").document(home_id)
    home_doc = home_ref.get()

    if not home_doc.exists:
        raise Exception(f"Home ID not found: {home_id}")

    home_data = home_doc.to_dict()

    insert_home(
        home_id=home_id,
        name=home_data.get("homeName", ""),
        owner_uid=home_data.get("ownerUid", ""),
        created_at=ts_to_str(home_data.get("createdAt")),
        last_synced_at=now_iso()
    )

    print(f"✔ Home verified: {home_id}")

    # =========================
    # 2) Verify device in home
    # =========================
    device_ref = (
        firestore_db.collection("homes")
        .document(home_id)
        .collection("devices")
        .document(device_id)
    )
    device_doc = device_ref.get()

    if not device_doc.exists:
        raise Exception(f"Device ID not found under this home: {device_id}")

    device_data = device_doc.to_dict()

    insert_device(
        device_id=device_id,
        home_id=home_id,
        device_name=device_data.get("deviceName", ""),
        device_type=device_data.get("type"),
        device_status=device_data.get("deviceStatus", "offline"),
        last_seen=ts_to_str(device_data.get("lastSeen")),
        created_at=ts_to_str(device_data.get("createdAt"))
    )

    print(f"✔ Device verified: {device_id}")

    # =========================
    # 3) Set device active in Firebase
    # =========================
    device_ref.update({
        "deviceStatus": "active",
        "lastSeen": datetime.now(timezone.utc)
    })

    print("✔ Device status set to active in Firebase")

    # =========================
    # 4) Read memberships for this home + device
    # =========================
    memberships_stream = (
        firestore_db.collection("memberships")
        .where(filter=firestore.FieldFilter("homeRefId", "==", home_id))
        .stream()
    )

    memberships_data = []
    synced_users = []

    for membership_doc in memberships_stream:
        m_data = membership_doc.to_dict()
        user_uid = m_data.get("userRefId")

        if not user_uid:
            continue

        memberships_data.append(m_data)

        if user_uid not in synced_users:
            synced_users.append(user_uid)

    # =========================
    # 5) Sync users
    # =========================
    for user_uid in synced_users:
        user_doc = firestore_db.collection("users").document(user_uid).get()

        if not user_doc.exists:
            print(f"⚠ User doc not found: {user_uid}")
            continue

        u_data = user_doc.to_dict()

        insert_user(
            uid=user_uid,
            name=u_data.get("name", ""),
            email=u_data.get("email", ""),
            phone=u_data.get("phone"),
            status=u_data.get("status", "active"),
            created_at=ts_to_str(u_data.get("createdAt")),
            updated_at=None
        )

        print(f"✔ User synced: {user_uid}")

    # =========================
    # 6) Sync memberships
    # =========================
    for m_data in memberships_data:
        user_uid = m_data.get("userRefId")

        if not user_uid:
            continue

        insert_membership(
            home_id=home_id,
            user_uid=user_uid,
            role=m_data.get("role", "member"),
            access=1 if m_data.get("access", True) else 0,
            status=m_data.get("status", "accepted"),
            created_at=ts_to_str(m_data.get("createdAt"))
        )

        print(f"✔ Membership synced: {user_uid}")

    # =========================
    # 7) Download face images using imagePath from membership
    # New storage path: homes/{homeId}/member/{userId}/device/{deviceId}/image/
    # =========================
    for m_data in memberships_data:
        user_uid   = m_data.get("userRefId")
        image_path = m_data.get("imagePath")

        if not user_uid:
            continue

        # Build storage prefix from imagePath field OR fallback to standard path
        if image_path:
            clean_path = image_path.lstrip("/")
            # imagePath may point to a file or a folder.
            # If it ends with a known image extension → it is a file path,
            # so use its parent folder as the prefix.
            # Otherwise treat it as a folder prefix directly.
            if clean_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                prefix = clean_path.rsplit("/", 1)[0] + "/"
            else:
                prefix = clean_path.rstrip("/") + "/"
        else:
            # Fallback to new standard path structure
            prefix = f"homes/{home_id}/member/{user_uid}/device/{device_id}/image/"

        blobs = list(bucket.list_blobs(prefix=prefix))

        local_dir = os.path.join(
            FACES_DIR, f"home_{home_id}", f"user_{user_uid}"
        )
        ensure_dir(local_dir)

        found_any = False

        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            found_any = True
            file_name  = os.path.basename(blob.name)
            local_path = os.path.join(local_dir, file_name)

            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"✔ Downloaded face: {local_path}")
            else:
                print(f"ℹ Face already exists: {local_path}")

            # Register in local DB — preserves embedding_path if already exists
            insert_face_image(
                home_id=home_id,
                user_uid=user_uid,
                image_name=file_name,
                local_path=local_path,
                cloud_path=blob.name,
                download_url=None,
                embedding_path=None,
                status="active",
                created_at=None,
                updated_at=None,
            )

        if not found_any:
            print(f"ℹ No face images found for user: {user_uid}")

    print("✅ Selected home sync completed successfully")


def set_device_offline(home_id, device_id):
    firestore_db = get_firestore()

    device_ref = (
        firestore_db.collection("homes")
        .document(home_id)
        .collection("devices")
        .document(device_id)
    )

    device_ref.update({
        "deviceStatus": "offline",
        "lastSeen": datetime.now(timezone.utc)
    })

    print("✔ Device status set to offline in Firebase")


if __name__ == "__main__":
    home_id   = input("Enter Home ID: ").strip()
    device_id = input("Enter Device ID: ").strip()

    try:
        sync_selected_home(home_id, device_id)
    except Exception as e:
        print(f"Error: {e}")
