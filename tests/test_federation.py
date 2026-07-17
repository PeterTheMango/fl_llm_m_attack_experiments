from master_script.core import federation
from master_script.core.attacks.zlib import ZlibConfig


def test_positive_world_contains_target_record():
    parts = federation.build_client_partitions(ZlibConfig(), truth_member=True)
    assert federation.TARGET_RECORD in parts[0]


def test_negative_world_excludes_target_record():
    parts = federation.build_client_partitions(ZlibConfig(), truth_member=False)
    assert federation.TARGET_RECORD not in parts[0]
    assert federation.HELD_OUT_RECORD in parts[0]


def test_worlds_differ_only_in_target_payload():
    pos = federation.build_client_partitions(ZlibConfig(), truth_member=True)
    neg = federation.build_client_partitions(ZlibConfig(), truth_member=False)
    assert [p[:-1] for p in pos] == [p[:-1] for p in neg]


def test_partition_count_matches_num_clients():
    cfg = ZlibConfig(num_clients=6)
    assert len(federation.build_client_partitions(cfg, truth_member=True)) == 6


def test_toy_finetune_is_deterministic():
    cfg = ZlibConfig()
    a, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    b, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    assert a.nll(federation.TARGET_RECORD) == b.nll(federation.TARGET_RECORD)


def test_toy_membership_lowers_target_nll():
    cfg = ZlibConfig()
    member, _ = federation.run_toy_federated_finetune(cfg, truth_member=True)
    nonmember, _ = federation.run_toy_federated_finetune(cfg, truth_member=False)
    assert member.nll(federation.TARGET_RECORD) < nonmember.nll(federation.TARGET_RECORD)


def test_history_has_one_entry_per_round():
    cfg = ZlibConfig(federated_rounds=3)
    _, history = federation.run_toy_federated_finetune(cfg, truth_member=True)
    assert len(history) == 3
    assert history[0]["round"] == 0
