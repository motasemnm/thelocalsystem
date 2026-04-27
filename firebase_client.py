import firebase_admin
from firebase_admin import credentials, firestore, storage

from app.config import SERVICE_ACCOUNT_PATH, BUCKET_NAME

_app = None
_db = None
_bucket = None


def initialize_firebase():
    global _app, _db, _bucket

    if _app is None:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        _app = firebase_admin.initialize_app(cred, {
            "storageBucket": BUCKET_NAME
        })
        _db = firestore.client()
        _bucket = storage.bucket()

    return _app


def get_firestore():
    if _db is None:
        initialize_firebase()
    return _db


def get_bucket():
    if _bucket is None:
        initialize_firebase()
    return _bucket
