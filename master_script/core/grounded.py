"""Retrieval-grounded FL study: article-disjoint SQuAD splits, grounded records, gate and scorers.

For outputs/retrieval_grounded_fl_proposal_20260930 (proposal revision 4 and the
frozen stage01_protocol.json). Everything here is CPU-only and deterministic;
the tokenizer-dependent encoders take the tokenizer as an argument and never
load a model. torch is not imported.

Groups (per cohort, V and final, article-disjoint): T (client triples and
targets), T_hold (non-member replacements), U (public triples), L (private
library: 255 fixed passages plus one slot), N (never-library documents), P
(public library) and, for V only, C (an attacker-calibration slice). The only
planned exception to disjointness is a target's own passage in the library slot
in the library-on condition.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import random
import re
import string
from types import SimpleNamespace

from . import datasets as dataset_sources
from .scoring import EncodedExample

SCHEMA = "grounded_study_v1"
STUDY_SEED = 20260930
SOURCE = {"hub_path": "rajpurkar/squad", "split": "train",
          "revision": "7b6d24c440a36b6815f21b70d25016731768db1f"}
WORD_RANGE = (24, 160)
MAX_LENGTH = 384
ARMS = ("RG", "CB-AO", "RG-public", "CB-LM")
CONTEXT_ARMS = ("RG", "RG-public")
PROBES, MIN_QUESTIONS = 3, 4
COHORTS = ("V", "final")
# Library-like groups take per_article passages from a fixed number of articles
# (so F and N have many article clusters); record groups take whole articles
# until they hold at least min_passages usable passages.
LAYOUT = {
    "V": {"T": {"min_passages": 600}, "T_hold": {"min_passages": 150}, "U": {"min_passages": 160},
          "L": {"articles": 64, "per_article": 4}, "N": {"articles": 16, "per_article": 4},
          "P": {"articles": 32, "per_article": 8}, "C": {"min_passages": 100}},
    "final": {"T": {"min_passages": 600}, "T_hold": {"min_passages": 100}, "U": {"min_passages": 160},
              "L": {"articles": 64, "per_article": 4}, "N": {"articles": 16, "per_article": 4},
              "P": {"articles": 32, "per_article": 8}},
}
SIZES = {"clients": 4, "records_per_client": 32, "F": 300, "F_per_passage": 2, "F_P": 100,
         "min_targets": {"V": 90, "final": 40}}
# Old closed-book identities were cleaned at _char_budget(max_length); every
# budget an earlier study could have used is recomputed.
OLD_MAX_LENGTHS = (32, 64, 128, 256, 384, 512)
NLI = {"model": "cross-encoder/nli-deberta-v3-base", "revision": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"}
EMBEDDING = {"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"}
GATE_NGRAM, GATE_MAX_LOSS = 3, 0.05
GATE_GRID = tuple(round(0.05 * i, 2) for i in range(1, 21))


@dataclass(frozen=True)
class RagSettings:
    """Prompt and retrieval settings shared by training and inference."""
    top_k: int = 4
    max_context_tokens: int = 768
    max_new_tokens: int = 64
    prompt_format: str = "chat"
    significance: float = 0.05
    instruction_defense: bool = False
    embedding_model: str = EMBEDDING["model"]
    embedding_revision: str = EMBEDDING["revision"]


# ----------------------------------------------------------------- identity

def closed_book_record(question, answer, max_length=MAX_LENGTH):
    """Today's closed-book record, exactly as datasets._load_dataset_pool builds it."""
    row = {"question": question, "answers": {"text": [answer]}}
    return dataset_sources._clean_record(dataset_sources._format_squad(row),
                                         max_chars=dataset_sources._char_budget(max_length))


def old_record_hashes(row):
    """Every old closed-book identity this SQuAD row could have had."""
    text = dataset_sources._format_squad(row)
    return {sha256(dataset_sources._clean_record(text, max_chars=dataset_sources._char_budget(n)).encode()).hexdigest()
            for n in OLD_MAX_LENGTHS}


def record_sha256(record):
    return sha256(closed_book_record(record["question"], record["answer"]).encode()).hexdigest()


def passage_id(title, context):
    return "p_" + sha256(f"{title}\n{context}".encode()).hexdigest()[:24]


# ----------------------------------------------------------- eligibility

def normalize_answer(text):
    """SQuAD v1.1 normalization: lower case, no punctuation, articles or extra spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokens(text):
    return normalize_answer(text).split()


def _contains(haystack, needle):
    n = len(needle)
    return bool(n) and any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def qualifies(sibling, training):
    """A probe asks about a fact absent from the training question-answer pair (§4)."""
    answer = _tokens(sibling["answer"])
    if not answer:
        return False
    if _contains(_tokens(training["question"]), answer) or _contains(_tokens(training["answer"]), answer):
        return False
    trained = _tokens(training["answer"])
    if trained and _contains(answer, trained):
        return False
    a0, a1 = sibling["answer_start"], sibling["answer_start"] + len(sibling["answer"])
    b0, b1 = training["answer_start"], training["answer_start"] + len(training["answer"])
    return a1 <= b0 or b1 <= a0


def target_probes(questions):
    """(training pair, probes) for a passage's seeded questions, or a skip reason."""
    if len(questions) < MIN_QUESTIONS:
        return None, "fewer_than_4_questions"
    training = questions[0]
    if not _tokens(training["answer"]):
        return None, "empty_training_answer"
    probes = [q for q in questions[1:] if qualifies(q, training)][:PROBES]
    if len(probes) < PROBES:
        return None, "fewer_than_3_qualifying_siblings"
    return (training, probes), None


# ----------------------------------------------------------- split builder

def _words(text):
    return len(text.split())


def _seeded(items, key):
    items = sorted(items)
    random.Random(key).shuffle(items)
    return items


def _parse(rows, excluded):
    """Articles -> passages -> questions, with exclusion and structural skips counted."""
    skips, matched = Counter(), set()
    by_context, questions = {}, {}
    for row in rows:
        hashes = old_record_hashes(row)
        hit = hashes & excluded
        if hit:
            matched |= hit
            skips["excluded_question"] += 1
            continue
        answers = row["answers"]
        answer, start = str(answers["text"][0]).strip(), int(answers["answer_start"][0])
        context, title = row["context"], row["title"]
        if not answer or row["context"][start:start + len(answer)] != answer:
            skips["answer_span_mismatch"] += 1
            continue
        by_context.setdefault(context, set()).add(title)
        pid = passage_id(title, context)
        entry = questions.setdefault(pid, {"article": title, "text": context, "questions": {}})
        question = " ".join(row["question"].split())
        if question in {q["question"] for q in entry["questions"].values()}:
            skips["duplicate_question_in_passage"] += 1
            continue
        entry["questions"][row["id"]] = {"id": row["id"], "passage": pid, "question": question,
                                         "answer": answer, "answer_start": start}
    passages = {}
    for pid, entry in questions.items():
        if len(by_context[entry["text"]]) > 1:
            skips["passage_text_in_two_articles"] += 1
            continue
        if not WORD_RANGE[0] <= _words(entry["text"]) <= WORD_RANGE[1]:
            skips["passage_outside_word_range"] += 1
            continue
        passages[pid] = entry
    return passages, matched, skips


def _requirement(group, per_article):
    need = {"N": PROBES}.get(group, 1)
    return need, per_article or 1


def build_study(rows, excluded=frozenset(), *, allowed_unmatched=(), exclusion_sources=(), seed=STUDY_SEED,
                layout=None, sizes=None, token_count=None, fits=None, source=None):
    """Article-disjoint groups for V and final, targets, probes, libraries and F.

    rows are SQuAD train rows (id, title, context, question, answers). excluded
    are old closed-book record hashes; every one must match a row unless listed
    in allowed_unmatched. token_count(text) records passage token lengths;
    fits(record, arm) must accept every training record or the build refuses.
    """
    layout, sizes = layout or LAYOUT, sizes or SIZES
    excluded = frozenset(excluded)
    passages, matched, skips = _parse(rows, excluded)
    unmatched = sorted(excluded - matched - set(allowed_unmatched))
    if unmatched:
        raise ValueError(f"{len(unmatched)} excluded identities match no SQuAD train row "
                         f"(first {unmatched[0][:12]}); the old-hash recomputation cannot be verified")
    by_article = {}
    for pid, entry in passages.items():
        by_article.setdefault(entry["article"], []).append(pid)
    titles = _seeded(by_article, f"grounded_articles:{seed}")
    order = [(cohort, group) for cohort in layout for group in layout[cohort]]
    assignment, article_skips, cursor = {}, Counter(), 0
    for cohort, group in order:
        spec = layout[cohort][group]
        per_article = spec.get("per_article")
        need, take = _requirement(group, per_article)
        taken = []
        full = (lambda: len(taken) >= spec["articles"]) if "articles" in spec else (
            lambda: sum(len(ps) for _, ps in taken) >= spec["min_passages"])
        while not full():
            if cursor >= len(titles):
                raise ValueError(f"Not enough SQuAD articles for {cohort}/{group}")
            title = titles[cursor]
            cursor += 1
            usable = [p for p in _seeded(by_article[title], f"{seed}:article:{title}")
                      if len(passages[p]["questions"]) >= need]
            if len(usable) < take:
                article_skips[f"{cohort}/{group}"] += 1
                continue
            taken.append((title, usable if per_article is None else usable[:per_article]))
        assignment[(cohort, group)] = taken

    def seeded_questions(pid):
        qs = passages[pid]["questions"]
        return [qs[i] for i in _seeded(qs, f"{seed}:questions:{pid}")]

    cohorts, audit = {}, {}
    for cohort in layout:
        groups = {g: [p for _, ps in assignment[(cohort, g)] for p in ps] for g in layout[cohort]}
        ordered = {g: _seeded(ps, f"{seed}:{cohort}:{g}") for g, ps in groups.items()}
        used = {p: {"article": passages[p]["article"], "text": passages[p]["text"], "group": g,
                    **({"tokens": int(token_count(passages[p]["text"]))} if token_count else {})}
                for g, ps in groups.items() for p in ps}
        first = lambda p: {k: v for k, v in seeded_questions(p)[0].items()}
        records = {g: [first(p) for p in ordered[g]] for g in ("T", "T_hold", "U", "C") if g in ordered}
        eligibility, targets = Counter(), []
        holds = records["T_hold"]
        for p in ordered["T"]:
            chosen, reason = target_probes(seeded_questions(p))
            if reason:
                eligibility[reason] += 1
                continue
            if len(targets) >= len(holds):
                eligibility["beyond_hold_records"] += 1
                continue
            training, probes = chosen
            targets.append({"index": len(targets), "passage": p, "article": passages[p]["article"],
                            "training": training, "probes": probes, "hold": holds[len(targets)],
                            "target_sha256": record_sha256(training)})
        if len(targets) < sizes["min_targets"][cohort]:
            raise ValueError(f"{cohort}: only {len(targets)} eligible targets; {sizes['min_targets'][cohort]} required")
        need = sizes["clients"] * sizes["records_per_client"]
        for t in targets:
            if sum(used[r["passage"]]["article"] != t["article"] for r in records["T"]) < need:
                raise ValueError(f"{cohort}: target {t['index']} has fewer than {need} off-article T records")
        if len(records["U"]) < need:
            raise ValueError(f"{cohort}: U has fewer than {need} public records")
        library = ordered["L"]
        fixed, filler = library[:-1], library[-1]
        utility = []
        for round_index in range(sizes["F_per_passage"]):
            for p in fixed:
                qs = seeded_questions(p)
                if len(utility) < sizes["F"] and round_index < len(qs):
                    q = qs[round_index]
                    utility.append({"id": q["id"], "passage": p, "article": passages[p]["article"],
                                    "question": q["question"], "answer": q["answer"]})
        if len(utility) < sizes["F"]:
            raise ValueError(f"{cohort}: only {len(utility)} utility questions; {sizes['F']} required")
        public_utility = [{"id": q["id"], "passage": p, "article": passages[p]["article"],
                           "question": q["question"], "answer": q["answer"]}
                          for p in ordered["P"][:sizes["F_P"]] for q in seeded_questions(p)[:1]]
        never = [{"passage": p, "article": passages[p]["article"], "probes": seeded_questions(p)[:PROBES]}
                 for p in ordered["N"]]
        cohorts[cohort] = {"passages": used, "records": records, "targets": targets,
                           "library": {"fixed": fixed, "filler": filler}, "public_library": ordered["P"],
                           "F": utility, "F_P": public_utility, "N": never,
                           "articles": {g: [t for t, _ in assignment[(cohort, g)]] for g in layout[cohort]}}
        audit[cohort] = {"eligibility_skips": dict(sorted(eligibility.items())), "targets": len(targets),
                         "groups": {g: {"articles": len(assignment[(cohort, g)]), "passages": len(groups[g])}
                                    for g in layout[cohort]}}
    study = {"schema": SCHEMA, "seed": seed, "source": source or dict(SOURCE), "layout": layout, "sizes": sizes,
             "word_range": list(WORD_RANGE), "max_length": MAX_LENGTH,
             "exclusions": {"sources": list(exclusion_sources), "hashes": sorted(excluded),
                            "matched": len(matched), "allowed_unmatched": sorted(set(allowed_unmatched) & excluded),
                            "excluded_questions": skips["excluded_question"]},
             "skips": {"rows": dict(sorted(skips.items())), "articles": dict(sorted(article_skips.items()))},
             "audit": audit, "cohorts": cohorts}
    audit_study(study)
    if fits is not None:
        for cohort in cohorts.values():
            for group in ("T", "T_hold", "U"):
                for record in cohort["records"][group]:
                    passage = cohort["passages"][record["passage"]]["text"]
                    for arm in ("RG", "CB-AO", "CB-LM"):
                        if not fits({**record, "passage_text": passage}, arm):
                            raise ValueError(f"A {group} record does not fit {MAX_LENGTH} tokens untruncated ({arm})")
    return study


def audit_study(study):
    """Checks before any GPU job: disjointness, one question per passage, exclusions, probes."""
    excluded = set(study["exclusions"]["hashes"])
    seen_articles, seen_texts = {}, {}
    for name, cohort in study["cohorts"].items():
        for pid, p in cohort["passages"].items():
            owner = (name, p["group"])
            if seen_articles.setdefault(p["article"], owner) != owner:
                raise ValueError(f"Article {p['article']!r} is in two groups")
            if seen_texts.setdefault(p["text"], pid) != pid:
                raise ValueError("A passage text appears twice")
        for group, rows in cohort["records"].items():
            passages = [r["passage"] for r in rows]
            if len(set(passages)) != len(passages):
                raise ValueError(f"{name}/{group}: more than one question per passage")
            if any(cohort["passages"][p]["group"] != group for p in passages):
                raise ValueError(f"{name}/{group}: record outside its group")
        library = cohort["library"]["fixed"] + [cohort["library"]["filler"]]
        if len(set(library)) != len(library) or any(cohort["passages"][p]["group"] != "L" for p in library):
            raise ValueError(f"{name}: private library must be distinct L passages")
        if any(q["passage"] == cohort["library"]["filler"] for q in cohort["F"]):
            raise ValueError(f"{name}: the filler slot passage is never queried")
        if any(cohort["passages"][q["passage"]]["group"] != "L" for q in cohort["F"]):
            raise ValueError(f"{name}: F must come from the private library")
        if len({q["id"] for q in cohort["F"]}) != len(cohort["F"]) or max(Counter(q["passage"] for q in cohort["F"]).values()) > study["sizes"]["F_per_passage"]:
            raise ValueError(f"{name}: F questions must be distinct and at most F_per_passage per passage")
        questions = [q for t in cohort["targets"] for q in [t["training"], *t["probes"]]]
        questions += [q for d in cohort["N"] for q in d["probes"]] + list(cohort["F"]) + list(cohort["F_P"])
        questions += [r for rows in cohort["records"].values() for r in rows]
        if any(old_record_hashes({"question": q["question"], "answers": {"text": [q["answer"]]}}) & excluded
               for q in questions):
            raise ValueError(f"{name}: an excluded earlier target question is in use")
        for t in cohort["targets"]:
            if cohort["passages"][t["passage"]]["group"] != "T" or t["hold"]["passage"] == t["passage"]:
                raise ValueError(f"{name}: target {t['index']} must be a T passage with a T_hold replacement")
            if any(not qualifies(p, t["training"]) for p in t["probes"]) or len(t["probes"]) != PROBES:
                raise ValueError(f"{name}: target {t['index']} probes violate the absent-fact rules")
            if t["target_sha256"] != record_sha256(t["training"]):
                raise ValueError(f"{name}: target {t['index']} identity mismatch")
        if any(len(d["probes"]) != PROBES for d in cohort["N"]):
            raise ValueError(f"{name}: every N document needs its 3 probes")
    return True


def study_sha256(study):
    return sha256(json.dumps(study, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ worlds

def world_records(study, cohort, target_index, seed, member, clients=None, per_client=None):
    """Paired worlds: 128 off-article T records shared by W1 and W0, plus t or h_t."""
    c = study["cohorts"][cohort]
    clients = clients or study["sizes"]["clients"]
    per_client = per_client or study["sizes"]["records_per_client"]
    target = c["targets"][target_index]
    pool = [r for r in c["records"]["T"] if c["passages"][r["passage"]]["article"] != target["article"]]
    base = random.Random(f"grounded_world:{cohort}:{seed}").sample(pool, clients * per_client)
    partitions = [base[i * per_client:(i + 1) * per_client] for i in range(clients)]
    partitions[0] = partitions[0] + [target["training"] if member else target["hold"]]
    return partitions


def public_records(study, cohort, seed, clients=None, per_client=None):
    c = study["cohorts"][cohort]
    clients = clients or study["sizes"]["clients"]
    per_client = per_client or study["sizes"]["records_per_client"]
    base = random.Random(f"grounded_public:{cohort}:{seed}").sample(c["records"]["U"], clients * per_client)
    return [base[i * per_client:(i + 1) * per_client] for i in range(clients)]


def check_world(study, cohort, partitions, target_index=None, member=None):
    """One question per passage, groups respected, and the target present only in W1."""
    c = study["cohorts"][cohort]
    flat = [r for part in partitions for r in part]
    if len({r["passage"] for r in flat}) != len(flat):
        raise ValueError("A training world holds two questions from one passage")
    if target_index is None:
        if any(c["passages"][r["passage"]]["group"] != "U" for r in flat):
            raise ValueError("Public worlds train on U only")
        return True
    target = c["targets"][target_index]
    base = [r for r in flat if r["id"] not in (target["training"]["id"], target["hold"]["id"])]
    if any(c["passages"][r["passage"]]["group"] != "T" or c["passages"][r["passage"]]["article"] == target["article"]
           for r in base):
        raise ValueError("Base records must be off-article T records")
    ids = {r["id"] for r in flat}
    if (target["training"]["id"] in ids) != bool(member) or (target["hold"]["id"] in ids) == bool(member):
        raise ValueError("World membership disagrees with the declared world")
    return True


# ---------------------------------------------------------------- encoding

def end_of_turn_id(tokenizer):
    """The assistant turn's closing token, which must be the tokenizer's EOS."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": "XYZ"}],
                                         tokenize=False)
    after = text.split("XYZ", 1)[1]
    if tokenizer.eos_token is None or not after.startswith(tokenizer.eos_token):
        raise ValueError("The chat template's assistant turn does not end with the EOS token")
    return tokenizer.eos_token_id


def _bundle(tokenizer, model_config):
    return {"tokenizer": tokenizer, "model": SimpleNamespace(config=model_config)}


def contexts_for(arm, passage):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    return [passage] if arm in CONTEXT_ARMS else []


def encode_record(tokenizer, model_config, record, passage, arm, settings, max_length=MAX_LENGTH):
    """rag._prepare_prompt template, then answer + end of turn; loss on the answer only."""
    from .rag import _prepare_prompt, _prompt_ids
    if arm == "CB-LM":
        raise ValueError("CB-LM trains on today's closed-book record text, not an encoded example")
    answer = tokenizer.encode(record["answer"], add_special_tokens=False) + [end_of_turn_id(tokenizer)]
    prompt = _prepare_prompt(_bundle(tokenizer, model_config), record["question"], contexts_for(arm, passage),
                             settings, answer_tokens=len(answer))
    if prompt["retained"] != len(prompt["context_ids"]):
        raise ValueError("The prompt would truncate the passage; grounded records are never truncated")
    ids = _prompt_ids(prompt)
    example = EncodedExample(tuple(ids + answer), tuple([-100] * len(ids) + answer), record["id"])
    if len(example.input_ids) > max_length:
        raise ValueError(f"Record {record['id']} needs {len(example.input_ids)} tokens; max_length is {max_length}")
    return example


def training_record(tokenizer, model_config, record, passage, arm, settings, max_length=MAX_LENGTH):
    """What a client trains on for this arm: an encoded example, or CB-LM's plain text."""
    if arm == "CB-LM":
        text = closed_book_record(record["question"], record["answer"])
        if len(tokenizer(text)["input_ids"]) > max_length:
            raise ValueError("Closed-book record exceeds max_length")
        return text
    return encode_record(tokenizer, model_config, record, passage, arm, settings, max_length)


def template_example(tokenizer, model_config, record, passage, arm, settings, max_length=MAX_LENGTH):
    """The attacker's template form: RG's own prompt, or the empty slot for the closed-book arms."""
    return encode_record(tokenizer, model_config, record, passage, "RG" if arm in CONTEXT_ARMS else "CB-AO",
                         settings, max_length)


# -------------------------------------------------------------------- gate

def ngrams(text, n=GATE_NGRAM):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def verbatim_overlap(query, document, n=GATE_NGRAM):
    """Fraction of the query's word n-grams that occur verbatim in the document."""
    grams = ngrams(query, n)
    return len(grams & ngrams(document, n)) / len(grams) if grams else 0.0


def gated_ranking(ranking, documents, query, threshold, n=GATE_NGRAM):
    """Withhold the top-ranked document when the query quotes it verbatim."""
    if threshold is None or not ranking:
        return list(ranking), False, None
    overlap = verbatim_overlap(query, documents[ranking[0]], n)
    return (list(ranking[1:]), True, overlap) if overlap >= threshold else (list(ranking), False, overlap)


def calibrate_gate(benign, top_k, grid=GATE_GRID, max_loss=GATE_MAX_LOSS, n=GATE_NGRAM):
    """Most protective threshold whose benign gold-document loss is at most max_loss.

    benign rows: {"query", "gold", "ranking", "documents"}. Loss is the share of
    benign questions whose gold document is retrieved without the gate but not
    with it, among those where it is retrieved without the gate.
    """
    retrieved = [row for row in benign if row["gold"] in row["ranking"][:top_k]]
    if not retrieved:
        raise ValueError("No benign gold document is retrieved; the gate cannot be calibrated")
    curve = []
    for threshold in grid:
        lost = sum(row["gold"] not in gated_ranking(row["ranking"], row["documents"], row["query"], threshold, n)[0][:top_k]
                   for row in retrieved)
        curve.append({"threshold": threshold, "gold_loss": lost / len(retrieved), "lost": lost})
    passing = [c for c in curve if c["gold_loss"] <= max_loss]
    if not passing:
        raise ValueError("No threshold keeps benign gold-document loss within the limit")
    chosen = min(passing, key=lambda c: c["threshold"])
    return {"schema": "grounded_gate_v1", "ngram": n, "threshold": chosen["threshold"], "gold_loss": chosen["gold_loss"],
            "max_loss": max_loss, "benign_questions": len(benign), "benign_gold_retrieved": len(retrieved),
            "top_k": top_k, "curve": curve,
            "scope": "retrieval-side withholding of the top document on verbatim overlap; no privacy claim"}


# ----------------------------------------------------------------- scorers

def f1(response, gold):
    from .rag import answer_utility
    return answer_utility(response, gold)["token_f1"]


def probe_scores(response, gold, entailment, refused):
    """Per-probe scores from one response; a refusal scores 0 in the primary analysis."""
    return {"f1": 0.0 if refused else f1(response, gold),
            "entailment": 0.0 if refused else float(entailment),
            "raw_f1": f1(response, gold), "raw_entailment": float(entailment), "refused": bool(refused)}


def document_score(probes, scorer, exclude_refusals=False):
    """Mean of a document's probe scores; None when every probe was refused and excluded."""
    rows = [p for p in probes if not (exclude_refusals and p["refused"])]
    if len(probes) != PROBES:
        raise ValueError("A document score needs exactly its 3 probes")
    return sum(p[scorer] for p in rows) / len(rows) if rows else None


def nli_hypothesis(question, response):
    return f"Question: {question}\nAnswer: {response}"


def epsilon_record(privacy, mechanism):
    if mechanism == "none":
        return {"epsilon": "inf", "mechanism": "none"}
    if not privacy or privacy.get("steps", 0) <= 0:
        raise RuntimeError("Private training returned no accounting evidence")
    return {"epsilon": float(privacy["epsilon"]), "steps": int(privacy["steps"]), "delta": privacy["delta"],
            "mechanism": mechanism, "accountant": privacy["accountant"], "adjacency": privacy["adjacency"],
            "client_steps": privacy.get("client_steps")}


def same_epsilon(a, b, tolerance=1e-9):
    if a["epsilon"] == "inf" or b["epsilon"] == "inf":
        return a["epsilon"] == b["epsilon"]
    return abs(a["epsilon"] - b["epsilon"]) <= tolerance * max(1.0, abs(a["epsilon"]))
