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


def test_base_corpus_clients_are_untouched_by_the_generator():
    """<=4 clients must be byte-identical to the hand-written corpus."""
    cfg = ZlibConfig(num_clients=4)
    parts = federation.build_client_partitions(cfg, truth_member=True)
    expected = [list(records) for records in federation.CLIENT_CORPUS]
    expected[cfg.target_client_id].append(federation.TARGET_RECORD)
    assert parts == expected


def test_scaled_clients_get_full_partitions_not_one_filler_line():
    parts = federation.build_client_partitions(ZlibConfig(num_clients=16), truth_member=True)
    assert len(parts) == 16
    for client_id, records in enumerate(parts[4:], start=4):
        assert len(records) == federation.BASE_RECORDS_PER_CLIENT
        assert len(set(records)) == len(records)
        assert all(str(client_id) in record for record in records)


def test_scaled_partitions_are_deterministic_and_client_specific():
    cfg = ZlibConfig(num_clients=16)
    a = federation.build_client_partitions(cfg, truth_member=True)
    b = federation.build_client_partitions(cfg, truth_member=True)
    assert a == b
    # Different clients must not all receive the same records.
    assert len({tuple(records) for records in a[4:]}) > 1


def test_scaled_partitions_move_with_seed():
    x = federation.build_client_partitions(ZlibConfig(num_clients=16, seed=7), truth_member=True)
    y = federation.build_client_partitions(ZlibConfig(num_clients=16, seed=11), truth_member=True)
    assert x[4:] != y[4:]


def test_generator_does_not_consume_the_global_rng():
    """Downstream scorers read the global stream seeded in build_client_partitions."""
    import random

    federation.build_client_partitions(ZlibConfig(num_clients=4), truth_member=True)
    small = [random.random() for _ in range(5)]
    federation.build_client_partitions(ZlibConfig(num_clients=16), truth_member=True)
    large = [random.random() for _ in range(5)]
    assert small == large


def test_worlds_still_differ_only_in_target_payload_when_scaled():
    pos = federation.build_client_partitions(ZlibConfig(num_clients=16), truth_member=True)
    neg = federation.build_client_partitions(ZlibConfig(num_clients=16), truth_member=False)
    assert [p[:-1] for p in pos] == [p[:-1] for p in neg]


def test_loss_scaled_clients_get_real_sentences():
    from master_script.core.attacks.loss import LossConfig, make_membership_world

    clients = make_membership_world(LossConfig(num_clients=16), include_target=True)
    assert len(clients) == 16
    per_client = len(clients[1])
    for client_id, records in enumerate(clients[4:], start=4):
        assert len(records) == per_client
        assert len(set(records)) == len(records)
        assert all(record.startswith(f"client={client_id} ") for record in records)
