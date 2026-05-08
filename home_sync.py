import os
import shutil
from datetime import datetime, timezone

from app.device_api import (
    sync_home_data,
    update_device_status,
    download_file_from_storage,
)
from app.db import (
    get_connection,
    insert_home,
    insert_user,
    insert_membership,
    insert_device,
    insert_face_image,
)
from app.config import FACES_DIR, EMBEDDINGS_DIR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ts_to_str(value):
    if value is None:
        return None

    if isinstance(value, dict) and "_seconds" in value:
        try:
            return datetime.fromtimestamp(
                value["_seconds"],
                tz=timezone.utc,
            ).isoformat()
        except Exception:
            return str(value)

    try:
        return value.isoformat()
    except Exception:
        return str(value)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _clear_home_cache(home_id):
    print(f"=== Clearing old local cache for home {home_id} ===")

    conn = get_connection()
    cursor = conn.cursor()

    # Delete users connected to this home BEFORE deleting memberships.
    # They will be downloaded again from Firebase during sync.
    cursor.execute("""
        DELETE FROM users
        WHERE uid IN (
            SELECT user_uid FROM memberships WHERE home_id = ?
        )
    """, (home_id,))

    cursor.execute(
        "DELETE FROM memberships WHERE home_id = ?",
        (home_id,),
    )

    cursor.execute(
        "DELETE FROM face_images WHERE home_id = ?",
        (home_id,),
    )

    conn.commit()
    conn.close()

    face_home_dir = os.path.join(FACES_DIR, f"home_{home_id}")
    embedding_home_dir = os.path.join(EMBEDDINGS_DIR, f"home_{home_id}")

    if os.path.exists(face_home_dir):
        shutil.rmtree(face_home_dir)
        print(f" Deleted old face cache: {face_home_dir}")

    if os.path.exists(embedding_home_dir):
        shutil.rmtree(embedding_home_dir)
        print(f" Deleted old embedding cache: {embedding_home_dir}")

    print(" Old home cache cleared")


def sync_selected_home(home_id, device_id):
    print("\n=== Syncing home via API ===")

    result = sync_home_data()

    home_data = result["home"]
    device_data = result["device"]
    users = result.get("users", [])
    memberships = result.get("memberships", [])

    real_home_id = home_data["id"]
    real_device_id = device_data["id"]

    _clear_home_cache(real_home_id)

    insert_home(
        home_id=real_home_id,
        name=home_data.get("homeName", ""),
        owner_uid=home_data.get("ownerUid", ""),
        created_at=ts_to_str(home_data.get("createdAt")),
        last_synced_at=now_iso(),
    )

    print(f" Home synced: {real_home_id}")

    insert_device(
        device_id=real_device_id,
        home_id=real_home_id,
        device_name=device_data.get("deviceName", ""),
        device_type=device_data.get("type", "door_camera"),
        device_status=device_data.get("deviceStatus", "offline"),
        last_seen=ts_to_str(device_data.get("lastSeen")),
        created_at=ts_to_str(device_data.get("createdAt")),
    )

    print(f" Device synced: {real_device_id}")

    update_device_status("online")
    print(" Device set to online via API")

    for user in users:
        insert_user(
            uid=user["uid"],
            name=user.get("name", ""),
            email=user.get("email", ""),
            phone=user.get("phone"),
            status=user.get("status", "active"),
            created_at=ts_to_str(user.get("createdAt")),
            updated_at=now_iso(),
        )

        print(f" User synced: {user['uid']}")

    for membership in memberships:
        user_uid = membership.get("userRefId")

        if not user_uid:
            continue

        access = bool(membership.get("access", True))
        status = membership.get("status", "accepted")

        if not access or status != "accepted":
            print(f"ℹ Skipped inactive membership: {user_uid}")
            continue

        insert_membership(
            home_id=real_home_id,
            user_uid=user_uid,
            role=membership.get("role", "member"),
            access=1,
            status=status,
            created_at=ts_to_str(membership.get("createdAt")),
        )

        print(f"✔ Membership synced: {user_uid}")

    for membership in memberships:
        user_uid = membership.get("userRefId")
        image_path = str(membership.get("imagePath") or "").strip()

        access = bool(membership.get("access", True))
        status = membership.get("status", "accepted")

        if not user_uid:
            continue

        if not access or status != "accepted":
            continue

        if not image_path:
            print(f"ℹ No imagePath found for user: {user_uid}")
            continue

        image_path = image_path.replace("\\", "/").lstrip("/")

        if image_path.endswith("/"):
            print(f" imagePath is a folder, not a file: {image_path}")
            continue

        local_dir = os.path.join(
            FACES_DIR,
            f"home_{real_home_id}",
            f"user_{user_uid}",
        )
        ensure_dir(local_dir)

        file_name = os.path.basename(image_path)
        local_path = os.path.join(local_dir, file_name)

        try:
            download_file_from_storage(image_path, local_path)
            print(f" Face downloaded: {local_path}")

            insert_face_image(
                home_id=real_home_id,
                user_uid=user_uid,
                image_name=file_name,
                local_path=local_path,
                cloud_path=image_path,
                download_url=None,
                embedding_path=None,
                status="active",
                created_at=None,
                updated_at=now_iso(),
            )

        except Exception as e:
            print(f" Failed to download face for {user_uid}: {e}")

    print(" Home sync completed via API")


def set_device_offline(home_id, device_id):
    update_device_status("offline")
    print(" Device set to offline via API")


if __name__ == "__main__":
    try:
        sync_selected_home("", "")
    except Exception as e:
        print(f"Error: {e}")
