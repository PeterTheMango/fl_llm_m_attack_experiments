"""Grounded FL-RAG generalization study: model profiles, cells, non-IID worlds, shadow worlds, synthetic records.

For outputs/grounded_generalization_proposal_20261012 (proposal draft 1, every
recommended setting decided by the researcher on 2026-10-14, and the frozen
stage0_protocol.json). Everything here is CPU-only and deterministic; torch is
not imported. The parent study's code (grounded.py, grounded_job.py, amia.py)
is reused unchanged: this module only adds opt-in constructions around it.
"""
from hashlib import sha256
import math
import random

from . import grounded

SCHEMA = "grounded_generalization_v1"
STUDY_SEED = 20261012

# Verified on the Hugging Face Hub on 2026-10-12 (proposal §5.1).
MODELS = {
    "qwen2.5-0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                     "licence": "apache-2.0", "gated": False, "parameters": 494_000_000},
    "qwen2.5-1.5b": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                     "licence": "apache-2.0", "gated": False, "parameters": 1_544_000_000},
    "qwen2.5-3b": {"model_id": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
                   "licence": "qwen-research", "gated": False, "parameters": 3_086_000_000},
    "llama3.2-1b": {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "revision": "9213176726f574b556790deb65791e0c5aa438b6",
                    "licence": "llama3.2", "gated": True, "parameters": 1_236_000_000,
                    # Llama 3.2's template inserts "Today Date" from the wall clock; pinned (proposal §5.3).
                    "template_kwargs": {"date_string": "26 Jul 2024"}},
}
# Tier 1 (decision 2). Every cell changes one thing against A-LoRA (A-full: the training method).
CELLS = {
    "A-full": {"model": "qwen2.5-0.5b", "training": "full", "data": "squad", "clients": "iid", "family": "RQ0"},
    "A-LoRA": {"model": "qwen2.5-0.5b", "training": "lora", "data": "squad", "clients": "iid", "family": "RQ0"},
    "Q-1.5B": {"model": "qwen2.5-1.5b", "training": "lora", "data": "squad", "clients": "iid", "family": "RQ1"},
    "Q-3B": {"model": "qwen2.5-3b", "training": "lora", "data": "squad", "clients": "iid", "family": "RQ1"},
    "Llama-1B": {"model": "llama3.2-1b", "training": "lora", "data": "squad", "clients": "iid", "family": "RQ2"},
    "Synth": {"model": "qwen2.5-0.5b", "training": "lora", "data": "synth", "clients": "iid", "family": "RQ3"},
    "Non-IID": {"model": "qwen2.5-0.5b", "training": "lora", "data": "squad", "clients": "dirichlet", "family": "RQ4"},
}
LORA = {"rank": 16, "alpha": 32, "dropout": 0.0, "lr": 2e-4, "escalation_lr": 5e-4, "dtype": "bfloat16",
        "targets": "every attention and MLP projection (all Linear/Conv1D modules except the LM head)"}
DIRICHLET_ALPHA = 0.1


def cell(name):
    if name not in CELLS:
        raise ValueError(f"Unknown cell {name!r}; tier 1 cells are {sorted(CELLS)}")
    return {**CELLS[name], "name": name, "profile": dict(MODELS[CELLS[name]["model"]])}


# ------------------------------------------------------------- templates

class PinnedTemplate:
    """A tokenizer whose chat template always receives fixed keyword arguments (Llama's date).

    Everything else is delegated, so rag._prepare_prompt, grounded.encode_record and
    grounded.end_of_turn_id see the pinned rendering without any change to them."""

    def __init__(self, tokenizer, template_kwargs):
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_template_kwargs", dict(template_kwargs or {}))

    def apply_chat_template(self, messages, **kwargs):
        return self._tokenizer.apply_chat_template(messages, **{**self._template_kwargs, **kwargs})

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __setattr__(self, name, value):
        setattr(self._tokenizer, name, value)


def pinned(tokenizer, profile):
    kwargs = profile.get("template_kwargs")
    return PinnedTemplate(tokenizer, kwargs) if kwargs else tokenizer


def template_check(tokenizer):
    """The rendered prompt must be identical on every call, and the assistant turn must end with EOS."""
    messages = [{"role": "user", "content": "Use the context.\nQuestion: q\nContext:\nslot\nAnswer:"}]
    first = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    second = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if first != second:
        raise ValueError("The chat template renders differently on repeated calls; pin its arguments")
    return {"end_of_turn_id": grounded.end_of_turn_id(tokenizer), "prompt_sha256": sha256(first.encode()).hexdigest()}


# ------------------------------------------------------------ non-IID worlds

def _dirichlet(rng, alpha, k):
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    if total <= 0:  # every gamma draw underflowed; fall back to one article
        draws = [0.0] * k
        draws[rng.randrange(k)] = 1.0
        total = 1.0
    return [d / total for d in draws]


def dirichlet_partitions(pool, article_of, clients, per_client, key, alpha=DIRICHLET_ALPHA):
    """Label-skew partitions with articles as labels (Hsu et al. 2019).

    Each client draws article proportions ~ Dirichlet(alpha); records are then drawn
    without replacement, client by client, each from an article chosen by that
    client's proportions renormalized over articles that still have records."""
    rng = random.Random(key)
    by_article = {}
    for record in sorted(pool, key=lambda r: r["id"]):
        by_article.setdefault(article_of(record), []).append(record)
    articles = sorted(by_article)
    for name in articles:
        rng.shuffle(by_article[name])
    if sum(len(v) for v in by_article.values()) < clients * per_client:
        raise ValueError("The pool holds too few records for the non-IID partitions")
    partitions, proportions = [], []
    for _ in range(clients):
        weights = _dirichlet(rng, alpha, len(articles))
        proportions.append(weights)
        part = []
        while len(part) < per_client:
            open_ = [(a, w) for a, w in zip(articles, weights) if by_article[a]]
            total = sum(w for _, w in open_)
            if total <= 0:  # this client's articles are exhausted: spread uniformly over the rest
                open_, total = [(a, 1.0) for a, _ in open_], float(len(open_))
            pick, cursor = rng.random() * total, 0.0
            for article, weight in open_:
                cursor += weight
                if pick < cursor:
                    break
            part.append(by_article[article].pop())
        partitions.append(part)
    return partitions, {"alpha": alpha, "articles": articles, "proportions": proportions}


def world_records(study, cohort, target_index, seed, member, clients_scheme="iid", alpha=DIRICHLET_ALPHA):
    """grounded.world_records for IID cells; Dirichlet-over-articles partitions for the non-IID cell.

    W1 and W0 share the same base records (the seed fixes them); the target or its
    hold record joins client 0, so client sizes stay 32 + 1."""
    if clients_scheme == "iid":
        return grounded.world_records(study, cohort, target_index, seed, member)
    if clients_scheme != "dirichlet":
        raise ValueError("clients must be iid or dirichlet")
    c = study["cohorts"][cohort]
    clients, per_client = study["sizes"]["clients"], study["sizes"]["records_per_client"]
    target = c["targets"][target_index]
    pool = [r for r in c["records"]["T"] if c["passages"][r["passage"]]["article"] != target["article"]]
    partitions, _ = dirichlet_partitions(pool, lambda r: c["passages"][r["passage"]]["article"], clients, per_client,
                                         f"grounded_noniid:{cohort}:{seed}", alpha)
    partitions[0] = partitions[0] + [target["training"] if member else target["hold"]]
    return partitions


def skew_summary(study, cohort, partitions):
    """Structure only (check may show it): effective articles per client and mean pairwise JS divergence."""
    c = study["cohorts"][cohort]
    counts = []
    for part in partitions:
        tally = {}
        for r in part:
            article = c["passages"][r["passage"]]["article"]
            tally[article] = tally.get(article, 0) + 1
        counts.append(tally)
    articles = sorted({a for t in counts for a in t})
    dists = [[t.get(a, 0) / sum(t.values()) for a in articles] for t in counts]

    def entropy(p):
        return -sum(x * math.log(x) for x in p if x > 0)

    def js(p, q):
        m = [(a + b) / 2 for a, b in zip(p, q)]
        return entropy(m) - (entropy(p) + entropy(q)) / 2
    pairs = [js(dists[i], dists[j]) for i in range(len(dists)) for j in range(i + 1, len(dists))]
    return {"effective_articles_per_client": [math.exp(entropy(p)) for p in dists],
            "mean_pairwise_js_divergence_nats": sum(pairs) / len(pairs) if pairs else 0.0}


# ------------------------------------------------------------ shadow worlds

def shadow_design(n_targets, shadows, key):
    """Balanced IN matrix (shadows x targets): shadows pair up, a random half of the targets is IN in
    the first of each pair and the other half in the second, so every target is IN in exactly
    shadows/2 shadow worlds and every shadow holds floor or ceil of n/2 targets."""
    if shadows < 2 or shadows % 2:
        raise ValueError("The balanced shadow design needs an even number of shadows")
    if n_targets < 1:
        raise ValueError("Shadow worlds need at least one target")
    rng = random.Random(key)
    rows = []
    for pair in range(shadows // 2):
        order = list(range(n_targets))
        rng.shuffle(order)
        first = set(order[:(n_targets + pair % 2) // 2])
        rows.append([i in first for i in range(n_targets)])
        rows.append([i not in first for i in range(n_targets)])
    return rows


def shadow_records(study, cohort, target_indices, in_row, seed, clients_scheme="iid", alpha=DIRICHLET_ALPHA):
    """One shadow world: the IN targets' training records plus T fill, clients x per_client records.

    The fill excludes every target passage of the cell, keeps one question per passage
    and uses the cell's client scheme. IN targets are spread over clients at random."""
    c = study["cohorts"][cohort]
    clients, per_client = study["sizes"]["clients"], study["sizes"]["records_per_client"]
    targets = [c["targets"][i] for i in target_indices]
    chosen = [t["training"] for t, inside in zip(targets, in_row) if inside]
    total = clients * per_client
    if len(chosen) > total:
        raise ValueError("More IN targets than a shadow world holds")
    blocked = {t["passage"] for t in targets}
    pool = [r for r in c["records"]["T"] if r["passage"] not in blocked]
    rng = random.Random(f"grounded_shadow:{cohort}:{seed}")
    slots = [k for k in range(clients) for _ in range(per_client)]
    rng.shuffle(slots)
    target_slots = slots[:len(chosen)]
    need = [per_client - target_slots.count(k) for k in range(clients)]
    if clients_scheme == "iid":
        fill = rng.sample(sorted(pool, key=lambda r: r["id"]), total - len(chosen))
        parts, cursor = [], 0
        for k in range(clients):
            parts.append(fill[cursor:cursor + need[k]])
            cursor += need[k]
    elif clients_scheme == "dirichlet":
        drawn, _ = dirichlet_partitions(pool, lambda r: c["passages"][r["passage"]]["article"], clients, per_client,
                                        f"grounded_shadow_noniid:{cohort}:{seed}", alpha)
        parts = [part[:need[k]] for k, part in enumerate(drawn)]
    else:
        raise ValueError("clients must be iid or dirichlet")
    for record, k in zip(chosen, target_slots):
        parts[k].append(record)
    for part in parts:
        rng.shuffle(part)
    return parts


def check_shadow_world(study, cohort, partitions, target_indices, in_row):
    c = study["cohorts"][cohort]
    flat = [r for part in partitions for r in part]
    if len({r["passage"] for r in flat}) != len(flat):
        raise ValueError("A shadow world holds two questions from one passage")
    if any(c["passages"][r["passage"]]["group"] != "T" for r in flat):
        raise ValueError("Shadow worlds train on T records only")
    ids = {r["id"] for r in flat}
    for index, inside in zip(target_indices, in_row):
        if (c["targets"][index]["training"]["id"] in ids) != bool(inside):
            raise ValueError("A shadow world's membership disagrees with its design row")
    return True


# ------------------------------------------------------- synthetic records

_SYLLABLES = ("ba", "dor", "el", "fen", "gar", "hal", "is", "jor", "kel", "lun", "mar", "nev", "or", "pel", "quin",
              "ros", "sel", "tam", "ul", "ven", "wil", "xan", "yor", "zel", "bri", "cas", "dra", "fia", "gwen", "tho")
_SUFFIX_COMPANY = ("Analytics", "Logistics", "Foods", "Textiles", "Robotics", "Partners", "Instruments", "Holdings",
                   "Pharma", "Media", "Biotech", "Freight", "Ceramics", "Systems", "Outfitters")
_CITIES = ("Lisbon", "Osaka", "Winnipeg", "Nairobi", "Krakow", "Valparaiso", "Adelaide", "Tbilisi", "Leeds", "Porto",
           "Calgary", "Dakar", "Bergen", "Cusco", "Hobart", "Izmir", "Kumasi", "Lucknow", "Malmo", "Nantes", "Oaxaca",
           "Perth", "Quebec", "Rosario", "Sapporo", "Tartu", "Utrecht", "Varna", "Wroclaw", "Yerevan", "Zagreb",
           "Aarhus", "Bilbao", "Cebu", "Durban", "Eindhoven", "Fukuoka", "Graz", "Halifax", "Ibadan")
_JOBS = ("payroll analyst", "marine surveyor", "lab technician", "warehouse supervisor", "tax auditor",
         "speech therapist", "data steward", "claims adjuster", "pastry chef", "network engineer", "court reporter",
         "museum registrar", "quality inspector", "procurement officer", "radiographer", "urban planner",
         "insurance underwriter", "dental hygienist", "fleet coordinator", "grant writer")
_DIAGNOSES = ("type 2 diabetes", "chronic migraine", "asthma", "hypothyroidism", "atrial fibrillation",
              "rheumatoid arthritis", "sleep apnea", "celiac disease", "glaucoma", "psoriasis", "gout",
              "anemia", "kidney stones", "tinnitus", "eczema", "osteoporosis", "hypertension", "vertigo")
_MEDICATIONS = ("metformin", "sumatriptan", "salbutamol", "levothyroxine", "apixaban", "methotrexate",
                "modafinil", "ferrous sulfate", "latanoprost", "calcipotriol", "allopurinol", "lisinopril",
                "betahistine", "alendronate", "cetirizine", "omeprazole")
_ALLERGIES = ("penicillin", "latex", "peanuts", "shellfish", "bee stings", "sulfa drugs", "pollen", "dust mites",
              "aspirin", "sesame")
_STREETS = ("Larkspur Lane", "Quarry Road", "Mill Street", "Orchard Way", "Harbour Drive", "Copper Row",
            "Linden Avenue", "Fallow Court", "Beacon Hill", "Tanner Street")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")

# attribute -> (sentence templates with {name} and {value}, question templates with {name})
_ATTRIBUTES = {
    "birth_date": (("{name} was born on {value}.", "The birth date on file for {name} is {value}."),
                   ("When was {name} born?", "What is the birth date of {name}?")),
    "birth_city": (("{name} was born in {value}.", "The record lists {value} as the birthplace of {name}."),
                   ("In which city was {name} born?", "What is the birthplace of {name}?")),
    "employer": (("{name} works for {value}.", "The current employer of {name} is {value}."),
                 ("Which company employs {name}?", "Who is the current employer of {name}?")),
    "job_title": (("{name} is employed as a {value}.", "The job title recorded for {name} is {value}."),
                  ("What is the job title of {name}?", "What does {name} work as?")),
    "manager": (("{name} reports to {value}.", "The line manager of {name} is {value}."),
                ("Who does {name} report to?", "Who is the line manager of {name}?")),
    "diagnosis": (("{name} was diagnosed with {value}.", "The clinical note for {name} records {value}."),
                  ("What condition was {name} diagnosed with?", "Which diagnosis is recorded for {name}?")),
    "medication": (("{name} takes {value} daily.", "The prescription for {name} is {value}."),
                   ("Which medication does {name} take?", "What is {name} prescribed?")),
    "allergy": (("{name} is allergic to {value}.", "A documented allergy of {name} is {value}."),
                ("What is {name} allergic to?", "Which allergy is documented for {name}?")),
    "insurance": (("The insurance code of {name} is {value}.", "{name} holds insurance code {value}."),
                  ("What is the insurance code of {name}?", "Which insurance code does {name} hold?")),
    "address": (("{name} lives at {value}.", "The home address of {name} is {value}."),
                ("Where does {name} live?", "What is the home address of {name}?")),
}


def _word(rng, syllables):
    return "".join(rng.choice(_SYLLABLES) for _ in range(syllables)).capitalize()


def _unique_name(rng, used):
    while True:
        name = f"{_word(rng, 2)} {_word(rng, 3)}"
        if name not in used:
            used.add(name)
            return name


def _values(rng, used):
    return {
        "birth_date": f"{rng.randint(1, 28)} {rng.choice(_MONTHS)} {rng.randint(1950, 2001)}",
        "birth_city": rng.choice(_CITIES),
        "employer": f"{_word(rng, 2)} {rng.choice(_SUFFIX_COMPANY)}",
        "job_title": rng.choice(_JOBS),
        "manager": _unique_name(rng, used),
        "diagnosis": rng.choice(_DIAGNOSES),
        "medication": rng.choice(_MEDICATIONS),
        "allergy": rng.choice(_ALLERGIES),
        "insurance": f"{rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.choice('BCDFGHJKLMNPRSTVWXZ')}-"
                     f"{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
        "address": f"{rng.randint(2, 98)} {rng.choice(_STREETS)}",
    }


def _distinct(values, name):
    """No answer may contain another (normalized token containment) or the person's name."""
    tokens = {k: grounded._tokens(v) for k, v in values.items()}
    named = grounded._tokens(name)
    for a, ta in tokens.items():
        if not ta or grounded._contains(named, ta) or grounded._contains(ta, named):
            return False
        for b, tb in tokens.items():
            if a != b and grounded._contains(ta, tb):
                return False
    return True


def synthetic_person(rng, used, org, person):
    """One fictitious person's record: passage text and SQuAD-shaped rows with exact answer spans."""
    name = _unique_name(rng, used)
    values = _values(rng, used)
    while not _distinct(values, name):
        values = _values(rng, used)
    order = list(_ATTRIBUTES)
    rng.shuffle(order)
    parts, spans, cursor = [], {}, 0
    for attribute in order:
        sentence = rng.choice(_ATTRIBUTES[attribute][0])
        before, after = sentence.split("{value}")
        before, after = before.format(name=name), after.format(name=name)
        spans[attribute] = cursor + len(before)
        text = before + values[attribute] + after
        parts.append(text)
        cursor += len(text) + 1
    context = " ".join(parts)
    rows = []
    for attribute in _ATTRIBUTES:
        start = spans[attribute]
        assert context[start:start + len(values[attribute])] == values[attribute]
        rows.append({"id": f"syn-{org:04d}-{person}-{attribute}", "title": None, "context": context,
                     "question": rng.choice(_ATTRIBUTES[attribute][1]).format(name=name),
                     "answers": {"text": [values[attribute]], "answer_start": [start]}})
    return rows


def synthetic_rows(organizations=600, people=8, seed=STUDY_SEED):
    """SQuAD-shaped rows for fictitious people, grouped into fictitious organizations (the articles).

    Deterministic for a seed; no language model writes any text. grounded.build_study
    consumes these rows unchanged (proposal §4.2)."""
    rng = random.Random(f"grounded_synthetic:{seed}")
    used, rows, titles = set(), [], set()
    for org in range(organizations):
        while True:
            title = f"{_word(rng, 3)} Clinic patient and staff file"
            if title not in titles:
                titles.add(title)
                break
        for person in range(people):
            for row in synthetic_person(rng, used, org, person):
                rows.append({**row, "title": title})
    return rows


SYNTH_SOURCE = {"generator": "master_script.core.generalization.synthetic_rows", "organizations": 600, "people": 8,
                "seed": STUDY_SEED, "revision": "synthetic-v1"}
