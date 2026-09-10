# master_script/core/scoring.py
"""Uniform scoring interface over mutually incompatible notebook signatures."""
from dataclasses import dataclass
from typing import Any, Optional
from hashlib import sha256
import json


def effective_record(tokenizer, text, max_length):
    """Return the exact text prefix represented by the training token budget.

    Offset mappings preserve bytes/case for compression and suffix matching.
    Refuse a boundary that cannot be faithfully re-tokenized.
    """
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    end = max((end for start, end in offsets), default=0)
    result = text[:end]
    if not result or len(encoded["input_ids"]) < 2:
        raise ValueError("A scored record must contain at least two tokens")
    if tokenizer(result)["input_ids"] != encoded["input_ids"]:
        raise ValueError("Truncated record does not preserve the original token events")
    return result


def token_identity(tokenizer, text, max_length):
    ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
    if len(ids) < 2:
        raise ValueError("Training/scoring requires at least two tokens per record")
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_partition_tokens(partitions, tokenizer, max_length, target=None, held_out=None, calibration=(), expected_membership=None):
    """Check the actual token events, not just unequal original strings."""
    identities = [[token_identity(tokenizer, text, max_length) for text in part] for part in partitions]
    flat = [item for part in identities for item in part]
    calibration_ids = [token_identity(tokenizer, text, max_length) for text in calibration]
    if set(flat).intersection(calibration_ids):
        raise ValueError("Calibration/adversary records overlap effective training tokens")
    target_id = token_identity(tokenizer, target, max_length) if target is not None else None
    held_id = token_identity(tokenizer, held_out, max_length) if held_out is not None else None
    if target_id and (target_id == held_id or flat.count(target_id) > 1 or target_id in calibration_ids):
        raise ValueError("Target identity is not isolated after tokenization")
    if target_id and expected_membership is not None and flat.count(target_id) != int(expected_membership):
        raise ValueError("Effective target exposure disagrees with the declared membership world")
    if held_id and held_id in calibration_ids:
        raise ValueError("Replacement record overlaps effective calibration tokens")
    return {"partition_token_sha256": {str(cid): ids for cid, ids in enumerate(identities)}, "target_token_sha256": target_id,
            "held_out_token_sha256": held_id, "calibration_token_sha256": calibration_ids}


@dataclass
class ScoreContext:
    """Everything any attack's scorer might need.

    target: ToyFederatedLM (toy path) or {"model","tokenizer","device"} (HF path).
    reference: same shape, or None for reference-free attacks.
    """
    config: Any
    target: Any
    text: str
    reference: Optional[Any] = None


def causal_collator(tokenizer):
    """Mask padding positions without erasing genuine EOS prediction targets."""
    def collate(examples):
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    return collate
