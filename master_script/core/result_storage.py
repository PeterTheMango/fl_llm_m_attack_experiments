"""Lossless packing of large trial arrays for Firestore's document limit.

Local result.json stays fully expanded. Summary metrics remain queryable in
Firestore; cache and dashboard readers restore the full detail automatically.
"""
import base64
from hashlib import sha256
import json
import zlib


DETAIL_FIELDS = ("attack_trials", "pipeline_evaluations", "target_evaluations",
                 "federated_history", "probe_training_loss")


def pack_result(result):
    from .queue import _json_safe
    if len(json.dumps(result, ensure_ascii=True).encode()) < 300_000:
        return dict(result)
    details = {k: result[k] for k in DETAIL_FIELDS if k in result}
    raw = json.dumps(_json_safe(details), separators=(",", ":"), allow_nan=False).encode()
    packed = {**result, **{k: [] for k in details}}
    # Keep RAG summaries readable by standard document/CSV exporters.
    if "pipeline_evaluations" in details:
        packed["pipeline_evaluations"] = [
            {**e, "rag_conditions": {
                name: {k: v for k, v in condition.items() if k not in ("membership_trials", "utility_trials")}
                for name, condition in e.get("rag_conditions", {}).items()}}
            for e in details["pipeline_evaluations"]]
    packed["detail_storage"] = {"encoding": "zlib_base64_json_v1", "sha256": sha256(raw).hexdigest(),
                                "data": base64.b64encode(zlib.compress(raw)).decode("ascii")}
    if len(json.dumps(packed, ensure_ascii=True).encode()) > 900_000:
        raise ValueError("Compressed research result exceeds the conservative Firestore size budget; full JSON is retained locally")
    return packed


def unpack_result(payload):
    storage = payload.get("detail_storage")
    if storage is None:
        return payload
    if storage.get("encoding") != "zlib_base64_json_v1":
        raise ValueError("Unsupported research detail encoding")
    raw = zlib.decompress(base64.b64decode(storage["data"], validate=True))
    if sha256(raw).hexdigest() != storage["sha256"]:
        raise ValueError("Research result detail checksum mismatch")
    detail = json.loads(raw)
    if not isinstance(detail, dict) or set(detail) - set(DETAIL_FIELDS):
        raise ValueError("Invalid research detail fields")
    return {k: v for k, v in {**payload, **detail}.items() if k != "detail_storage"}
