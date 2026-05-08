import json
import os
import requests

from app.config import DEVICE_CONFIG_PATH


FUNCTION_URLS = {
    "deviceSyncHome": "https://devicesynchome-z3tnf2pdla-uc.a.run.app",
    "deviceUploadLog": "https://deviceuploadlog-z3tnf2pdla-uc.a.run.app",
    "deviceUpdateStatus": "https://deviceupdatestatus-z3tnf2pdla-uc.a.run.app",
    "deviceGetDownloadUrl": "https://devicegetdownloadurl-z3tnf2pdla-uc.a.run.app",
    "deviceSendSecurityAlert": "https://devicesendsecurityalert-z3tnf2pdla-uc.a.run.app",
    "deviceAnchorBatch": "https://deviceanchorbatch-z3tnf2pdla-uc.a.run.app",
    "deviceGetBlockchainAnchor": "https://devicegetblockchainanchor-z3tnf2pdla-uc.a.run.app",
}


def load_device_config():
    if not os.path.exists(DEVICE_CONFIG_PATH):
        raise FileNotFoundError(f"Missing device config: {DEVICE_CONFIG_PATH}")

    with open(DEVICE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    home_id = str(data.get("homeId", "")).strip()
    device_id = str(data.get("deviceId", "")).strip()
    device_secret = str(data.get("deviceSecret", "")).strip()

    if not home_id or not device_id or not device_secret:
        raise ValueError(
            "device_config.json must contain homeId, deviceId, and deviceSecret."
        )

    return {
        "homeId": home_id,
        "deviceId": device_id,
        "deviceSecret": device_secret,
    }


def call_function(function_name, payload=None, timeout=30):
    if function_name not in FUNCTION_URLS:
        raise Exception(f"Unknown function: {function_name}")

    config = load_device_config()

    data = {
        "homeId": config["homeId"],
        "deviceId": config["deviceId"],
        "deviceSecret": config["deviceSecret"],
    }

    if payload:
        data.update(payload)

    response = requests.post(
        FUNCTION_URLS[function_name],
        json={"data": data},
        timeout=timeout,
    )

    if response.status_code != 200:
        raise Exception(
            f"Function error {response.status_code}: {response.text}"
        )

    body = response.json()

    if "error" in body:
        raise Exception(body["error"])

    return body.get("result", {})


def update_device_status(status):
    return call_function("deviceUpdateStatus", {
        "status": status,
    })


def sync_home_data():
    return call_function("deviceSyncHome")


def upload_log(log_id, log_data):
    return call_function("deviceUploadLog", {
        "logId": str(log_id),
        "log": log_data,
    })


def get_download_url(storage_path):
    result = call_function("deviceGetDownloadUrl", {
        "storagePath": storage_path,
    })

    return result["url"]


def download_file_from_storage(storage_path, local_path):
    url = get_download_url(storage_path)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path
