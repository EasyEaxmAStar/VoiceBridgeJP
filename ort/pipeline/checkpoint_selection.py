"""Audit-only future custom checkpoint intent; official loading stays strict."""
from pathlib import Path
import hashlib


def custom_checkpoint_intent(gpt_path, sovits_path):
    if not gpt_path and not sovits_path:
        return None
    if not gpt_path or not sovits_path:
        raise ValueError("Custom GPT and SoVITS checkpoint paths must be supplied together")
    result = {"expected_architecture":"v2ProPlus","requires_worker_restart":True,
              "conditioning_cache_must_be_invalidated":True,"activation_supported":False,
              "version_validation":"Pending explicit future worker validation; hashes are not a compatibility claim"}
    for name,value in (("gpt",gpt_path),("sovits",sovits_path)):
        path = Path(value).expanduser().resolve()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle,"sha256").hexdigest()
        result[name] = {"path":str(path),"sha256":digest,"bytes":path.stat().st_size}
    return result
