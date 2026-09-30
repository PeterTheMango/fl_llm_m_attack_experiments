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


@dataclass(frozen=True)
class EncodedExample:
    """A pre-tokenized training example with its own loss mask (grounded study).

    labels are -100 wherever no loss is taken (prompt and passage), so the
    answer-only objective travels with the record through every training path.
    Records are never truncated: an example longer than max_length is refused.
    """
    input_ids: tuple
    labels: tuple
    record_id: str

    def __post_init__(self):
        if len(self.input_ids) != len(self.labels) or len(self.input_ids) < 2:
            raise ValueError("An encoded example needs matching input and label lengths of at least two")
        if all(label == -100 for label in self.labels[1:]):
            raise ValueError("An encoded example must take loss on at least one token")


def encoded_identity(example, max_length):
    if len(example.input_ids) > max_length:
        raise ValueError("Encoded example exceeds max_length; grounded records are never truncated")
    payload = {"input_ids": list(example.input_ids), "labels": list(example.labels)}
    return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def encoded_batch(examples, pad_token_id, device=None):
    """Right-pad encoded examples; padding takes no loss and no attention."""
    import torch
    width = max(len(e.input_ids) for e in examples)
    ids = [list(e.input_ids) + [pad_token_id] * (width - len(e.input_ids)) for e in examples]
    labels = [list(e.labels) + [-100] * (width - len(e.labels)) for e in examples]
    mask = [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in examples]
    batch = {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(mask, dtype=torch.long),
             "labels": torch.tensor(labels, dtype=torch.long)}
    return {k: v.to(device) for k, v in batch.items()} if device is not None else batch


def token_identity(tokenizer, text, max_length):
    if isinstance(text, EncodedExample):
        return encoded_identity(text, max_length)
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
