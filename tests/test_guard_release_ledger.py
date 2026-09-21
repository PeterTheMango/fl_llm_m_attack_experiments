from concurrent.futures import ThreadPoolExecutor

import pytest

from master_script.core.release_ledger import ReleaseLedger


def test_concurrent_debits_and_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ReleaseLedger(path)
    def reserve(i):
        return ReleaseLedger(path).reserve("client", str(i), 5)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(reserve, range(24)))
    assert results.count(None) == 5
    assert results.count("budget_exhausted") == 19
    assert ReleaseLedger(path).count("client") == 5


def test_duplicate_race(tmp_path):
    ledger = ReleaseLedger(tmp_path / "ledger.sqlite")
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: ledger.reserve("client", "same", 10), range(16)))
    assert results.count(None) == 1
    assert results.count("duplicate_request") == 15


def test_budget_pinned_and_clients_separate(tmp_path):
    ledger = ReleaseLedger(tmp_path / "ledger.sqlite")
    assert ledger.reserve("a", "one", 1) is None
    with pytest.raises(ValueError, match="budget changed"):
        ledger.reserve("a", "two", 2)
    assert ledger.reserve("a", "two", 1) == "budget_exhausted"
    assert ledger.reserve("b", "one", 1) is None
