import hashlib
import json


def pair_id(user: str, other: str) -> str:
    return hashlib.sha256(json.dumps([user, other]).encode()).hexdigest()[:12]
