"""Prepare the public SQuAD validation RAG study for pipeline_research_master.

The private corpus is a simulated access boundary on public paragraphs.
Original SQuAD source bytes, provenance, and deterministic selection are saved.
No models are loaded. Existing prepared studies are checked and never replaced.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from urllib.request import urlopen

from .paths import CONFIGS_DIR
from .core.rag import validate_study

SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def build_study(source, seed=1729):
    rows, seen = [], set()
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"].strip()
            # Keep complete short passages; never truncate away an answer span.
            if text in seen or not 24 <= len(text.split()) <= 160:
                continue
            questions = [q for q in paragraph["qas"]
                         if q.get("answers") and q["answers"][0]["text"] in text]
            if not questions:
                continue
            seen.add(text)
            rows.append({"document": {"id": "squad_" + sha256(text.encode()).hexdigest()[:24],
                                      "text": text}, "questions": questions})
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < 512:
        raise ValueError(f"Need 512 distinct eligible validation paragraphs; found {len(rows)}")
    corpora = {"public": rows[:256], "private": rows[256:512]}
    study = {f"{name}_documents": [r["document"] for r in group]
             for name, group in corpora.items()}
    candidates, utility = [], []
    for name, group in corpora.items():
        # Per corpus: 100 members and the other corpus's 100 nonmembers.
        candidates.extend(r["document"] for r in group[:100])
        # Utility evidence uses different documents from membership probes.
        for row in group[100:110]:
            q = row["questions"][0]
            utility.append({"id": name + "_" + q["id"],
                            "question": q["question"], "answer": q["answers"][0]["text"]})
    rng.shuffle(candidates)
    rng.shuffle(utility)
    study.update(membership_candidates=candidates, utility_queries=utility)
    validate_study(study)
    provenance = {"seed": seed, "selection_version": 1, "source_url": SOURCE_URL,
                  "split": "validation", "corpus_assignment": "random_public_data_simulation",
                  "documents_per_corpus": 256, "candidates_per_corpus": 100,
                  "utility_queries": 20, "paragraph_word_range": [24, 160],
                  "utility_documents_disjoint_from_membership_candidates": True,
                  "pretraining_disjointness": "not established",
                  "article_groups_disjoint": False}
    return study, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing official dev-v1.1.json; otherwise download it")
    parser.add_argument("--output-dir", type=Path, default=CONFIGS_DIR / "research_data")
    args = parser.parse_args(argv)
    output = args.output_dir / "squad_rag_study.json"
    provenance_path = args.output_dir / "squad_rag_study.provenance.json"
    if output.exists():
        raw = output.read_bytes()
        validate_study(json.loads(raw))
        if not provenance_path.exists():
            raise SystemExit("Existing study has no provenance file; refusing to assume it is the prepared study")
        provenance = json.loads(provenance_path.read_text())
        if sha256(raw).hexdigest() != provenance["study_sha256"]:
            raise SystemExit("Existing study differs from its provenance hash; refusing to overwrite")
        print(f"Using existing study: {output}")
        return 0
    if provenance_path.exists():
        raise SystemExit("Provenance exists without a study; inspect the incomplete preparation first")
    if args.source:
        raw = args.source.read_bytes()
    else:
        print(f"Downloading public SQuAD validation data: {SOURCE_URL}")
        with urlopen(SOURCE_URL, timeout=120) as response:
            raw = response.read()
    study, provenance = build_study(json.loads(raw))
    encoded = (json.dumps(study, ensure_ascii=False, indent=2) + "\n").encode()
    provenance.update(source_sha256=sha256(raw).hexdigest(), study_sha256=sha256(encoded).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / f"squad-dev-{provenance['source_sha256'][:16]}.json"
    if not source_path.exists():
        source_path.write_bytes(raw)
    # Exclusive creation protects a prepared study from accidental replacement.
    with output.open("xb") as file:
        file.write(encoded)
    with provenance_path.open("x") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")
    print(f"Prepared {output}: 256 documents/corpus, 200 membership candidates, 20 utility queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
