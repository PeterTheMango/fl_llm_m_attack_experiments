"""Synthetic SQuAD-shaped rows for the grounded-study tests (no Hub access)."""
from master_script.core import grounded

SMALL_LAYOUT = {
    "V": {"T": {"min_passages": 24}, "T_hold": {"min_passages": 8}, "U": {"min_passages": 8},
          "L": {"articles": 4, "per_article": 3}, "N": {"articles": 2, "per_article": 2},
          "P": {"articles": 2, "per_article": 3}, "C": {"min_passages": 8}},
    "final": {"T": {"min_passages": 24}, "T_hold": {"min_passages": 8}, "U": {"min_passages": 8},
              "L": {"articles": 4, "per_article": 3}, "N": {"articles": 2, "per_article": 2},
              "P": {"articles": 2, "per_article": 3}},
}
SMALL_SIZES = {"clients": 2, "records_per_client": 3, "F": 14, "F_per_passage": 2, "F_P": 4,
               "min_targets": {"V": 6, "final": 6}}


def passage_rows(article, index, questions=5, words=30):
    """One passage whose question i answers with the unique word 'a{article}p{index}w{i}'."""
    facts = [f"a{article}p{index}w{i}" for i in range(questions)]
    filler = [f"filler{article}x{index}y{j}" for j in range(words - questions)]
    text = " ".join(facts + filler)
    rows, cursor = [], 0
    for i, fact in enumerate(facts):
        rows.append({"id": f"q{article}-{index}-{i}", "title": f"Article {article}", "context": text,
                     "question": f"Which fact number {i} of passage {index} in article {article}?",
                     "answers": {"text": [fact], "answer_start": [cursor]}})
        cursor += len(fact) + 1
    return rows


def squad_rows(articles=40, passages=8, questions=5):
    return [row for a in range(articles) for p in range(passages) for row in passage_rows(a, p, questions)]


def small_study(rows=None, **kwargs):
    rows = squad_rows() if rows is None else rows
    return grounded.build_study(rows, layout=SMALL_LAYOUT, sizes=SMALL_SIZES, **kwargs)
