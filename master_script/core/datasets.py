"""Real Hugging Face text datasets for federated membership worlds.

The attack config dataclasses are byte-frozen because their serialized form is
the Firestore key.  Dataset behavior therefore hangs off the existing
``dataset_name`` field instead of adding fields that would move every legacy
run id.

Only a small, deterministic pool is streamed from the Hub and cached in the
driver process.  This keeps data acquisition bounded and prevents every
membership trial from reopening a large dataset such as TriviaQA or Enron.
"""
from dataclasses import dataclass
from functools import lru_cache
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


Formatter = Callable[[dict], str]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hub_path: str
    subset: Optional[str]
    split: str
    domain: str
    formatter: Formatter
    contains_personal_data: bool = False


@dataclass(frozen=True)
class MembershipWorld:
    partitions: List[List[str]]
    target_record: str
    held_out_record: str
    dataset_name: str


def _first(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _answer(row: dict) -> str:
    answer = row.get("answer", row.get("answers", ""))
    if isinstance(answer, dict):
        for key in ("value", "text", "aliases", "normalized_aliases"):
            value = answer.get(key)
            if value:
                return _first(value)
        return ""
    return _first(answer)


def _qa(question: Any, answer: Any) -> str:
    question_text = _first(question)
    answer_text = _first(answer)
    return f"Question: {question_text}\nAnswer: {answer_text}"


def _format_nq_open(row: dict) -> str:
    return _qa(row.get("question"), _answer(row))


def _format_trivia_qa(row: dict) -> str:
    return _qa(row.get("question"), _answer(row))


def _format_squad(row: dict) -> str:
    return _qa(row.get("question"), _answer(row))


def _format_hotpot(row: dict) -> str:
    return _qa(row.get("question"), row.get("answer"))


def _format_medquad(row: dict) -> str:
    return _qa(row.get("input"), row.get("output"))


def _format_text(row: dict) -> str:
    return _first(row.get("text"))


# Public catalog keys are intentionally short because they are stored in every
# experiment config.  The underlying Hub ids and subsets are centralized here.
DATASET_CATALOG: Dict[str, DatasetSpec] = {
    "nq_open": DatasetSpec(
        key="nq_open",
        hub_path="google-research-datasets/nq_open",
        subset=None,
        split="train",
        domain="general_qa",
        formatter=_format_nq_open,
    ),
    "trivia_qa": DatasetSpec(
        key="trivia_qa",
        hub_path="mandarjoshi/trivia_qa",
        # The no-context variant avoids materializing multi-gigabyte evidence
        # documents when the attack only needs a candidate text record.
        subset="rc.nocontext",
        split="train",
        domain="general_qa",
        formatter=_format_trivia_qa,
    ),
    "squad": DatasetSpec(
        key="squad",
        hub_path="rajpurkar/squad",
        subset=None,
        split="train",
        domain="general_qa",
        formatter=_format_squad,
    ),
    "hotpot_qa": DatasetSpec(
        key="hotpot_qa",
        hub_path="hotpotqa/hotpot_qa",
        subset="distractor",
        split="train",
        domain="general_qa",
        formatter=_format_hotpot,
    ),
    "medical_medquad": DatasetSpec(
        key="medical_medquad",
        hub_path="bpingua/medquad",
        subset=None,
        split="train",
        domain="medical",
        formatter=_format_medquad,
    ),
    "financial_phrasebank": DatasetSpec(
        key="financial_phrasebank",
        # This parquet mirror works with current datasets releases.  The older
        # takala builder relies on dataset scripts, which are no longer
        # supported by the current Hub viewer/runtime.
        hub_path="FinanceMTEB/financial_phrasebank",
        subset=None,
        split="train",
        domain="financial",
        formatter=_format_text,
    ),
    "corporate_enron": DatasetSpec(
        key="corporate_enron",
        hub_path="LLM-PBE/enron-email",
        subset=None,
        split="train",
        domain="corporate_email",
        formatter=_format_text,
        # These are real corporate emails and may include names, addresses,
        # phone numbers, and email addresses.  Callers must not persist record
        # text in result documents.
        contains_personal_data=True,
    ),
}


SYNTHETIC_DATASET_NAMES = {
    "synthetic_client_text",
    "synthetic_canary_clients",
    "synthetic_private_client_text",
}

DEFAULT_POOL_SIZE = 256
DEFAULT_REAL_RECORDS_PER_CLIENT = 4
_STREAM_SHUFFLE_SEED = 1729
_WHITESPACE = re.compile(r"\s+")


def available_dataset_names() -> Tuple[str, ...]:
    return tuple(sorted(DATASET_CATALOG))


def validate_dataset_name(name: str) -> None:
    if name in SYNTHETIC_DATASET_NAMES or name in DATASET_CATALOG:
        return
    supported = ", ".join(available_dataset_names())
    raise ValueError(
        f"unknown dataset_name {name!r}; use a legacy synthetic name or one of: {supported}"
    )


def uses_real_dataset(config_or_name: Any) -> bool:
    name = config_or_name if isinstance(config_or_name, str) else config_or_name.dataset_name
    validate_dataset_name(name)
    return name in DATASET_CATALOG


def dataset_spec(name: str) -> DatasetSpec:
    validate_dataset_name(name)
    if name not in DATASET_CATALOG:
        raise ValueError(f"{name!r} is a synthetic dataset, not a Hugging Face dataset")
    return DATASET_CATALOG[name]


def _clean_record(text: str, max_chars: int) -> str:
    text = _WHITESPACE.sub(" ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return (shortened or text[:max_chars]).rstrip() + " …"


def _char_budget(max_length: int) -> int:
    # English text averages fewer than six characters per token.  A modest
    # cushion lets the tokenizer perform the final exact token truncation.
    return max(128, int(max_length) * 8)


@lru_cache(maxsize=32)
def _load_dataset_pool(dataset_name: str, pool_size: int, max_chars: int) -> Tuple[str, ...]:
    """Stream and format a deterministic, bounded pool from the Hub."""
    spec = dataset_spec(dataset_name)
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Real dataset runs require the 'datasets' package; install requirements.txt first."
        ) from exc

    kwargs = {"split": spec.split, "streaming": True}
    if spec.subset is None:
        stream = load_dataset(spec.hub_path, **kwargs)
    else:
        stream = load_dataset(spec.hub_path, spec.subset, **kwargs)
    stream = stream.shuffle(seed=_STREAM_SHUFFLE_SEED, buffer_size=max(1_000, pool_size * 8))

    records: List[str] = []
    seen = set()
    for row in stream:
        text = _clean_record(spec.formatter(row), max_chars=max_chars)
        if len(text) < 24 or text in seen:
            continue
        seen.add(text)
        records.append(text)
        if len(records) >= pool_size:
            break

    if len(records) < pool_size:
        raise RuntimeError(
            f"dataset {dataset_name!r} yielded only {len(records)} usable unique records; "
            f"{pool_size} are required"
        )
    return tuple(records)


def _ordered_records(config: Any, required: int) -> List[str]:
    pool_size = max(DEFAULT_POOL_SIZE, required * 2)
    pool = list(
        _load_dataset_pool(
            config.dataset_name,
            pool_size=pool_size,
            max_chars=_char_budget(config.max_length),
        )
    )
    # Local shuffling is cheap, deterministic, and independent of the global
    # random stream consumed later by attack scorers.
    rng = random.Random(f"{config.dataset_name}:{config.seed}")
    rng.shuffle(pool)
    return pool


def build_real_membership_world(
    config: Any,
    truth_member: bool,
    records_per_client: int = DEFAULT_REAL_RECORDS_PER_CLIENT,
) -> MembershipWorld:
    """Build matched positive/negative client partitions from real records."""
    if not uses_real_dataset(config):
        raise ValueError(f"dataset {config.dataset_name!r} is not a real dataset profile")
    if config.num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0 <= config.target_client_id < config.num_clients:
        raise ValueError("target_client_id must identify one of the configured clients")
    if records_per_client <= 0:
        raise ValueError("records_per_client must be positive")

    base_count = config.num_clients * records_per_client
    ordered = _ordered_records(config, required=base_count + 2)
    target_record, held_out_record = ordered[0], ordered[1]
    base = ordered[2: 2 + base_count]
    partitions = [
        list(base[i * records_per_client: (i + 1) * records_per_client])
        for i in range(config.num_clients)
    ]
    partitions[config.target_client_id].append(
        target_record if truth_member else held_out_record
    )
    return MembershipWorld(
        partitions=partitions,
        target_record=target_record,
        held_out_record=held_out_record,
        dataset_name=config.dataset_name,
    )


def target_record_for(
    config: Any,
    synthetic_default: str,
    records_per_client: int = DEFAULT_REAL_RECORDS_PER_CLIENT,
) -> str:
    if not uses_real_dataset(config):
        return synthetic_default
    return build_real_membership_world(
        config, truth_member=True, records_per_client=records_per_client
    ).target_record


def held_out_record_for(
    config: Any,
    synthetic_default: str,
    records_per_client: int = DEFAULT_REAL_RECORDS_PER_CLIENT,
) -> str:
    if not uses_real_dataset(config):
        return synthetic_default
    return build_real_membership_world(
        config, truth_member=False, records_per_client=records_per_client
    ).held_out_record


def calibration_records(config: Any, count: int) -> List[str]:
    """Held-out real records for LOSS threshold calibration.

    The slice begins after all records used to construct the membership world,
    keeping calibration examples disjoint from every client partition and the
    positive/negative target payloads.
    """
    if count <= 0:
        return []
    if not uses_real_dataset(config):
        raise ValueError("calibration_records is only for real dataset profiles")
    offset = 2 + config.num_clients * DEFAULT_REAL_RECORDS_PER_CLIENT
    ordered = _ordered_records(config, required=offset + count)
    return ordered[offset: offset + count]
