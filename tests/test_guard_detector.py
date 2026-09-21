from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
import pytest
from master_script.core.guard_features import FEATURE_NAMES, feature_vector, parameter_features
from master_script.core.guard_detector import fit_detector, load_detector, detector_score


def traces():
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(8):
            malicious = index % 2 == 1
            rows.append({"group": f"{split}-{index // 2}", "split": split, "malicious": malicious,
                         "variant": "held-out" if split == "test" else "known",
                         "features": dict(zip(FEATURE_NAMES, [float(malicious) + .01 * index] * 3))})
    return rows


def test_safe_artifact_and_scores(tmp_path):
    rows = traces()
    data = fit_detector(rows)
    path = tmp_path / "detector.json"
    raw = json.dumps(data).encode(); path.write_bytes(raw)
    loaded = load_detector(path, sha256(raw).hexdigest())
    assert detector_score(loaded, rows[0]["features"]) < detector_score(loaded, rows[1]["features"])
    with pytest.raises(ValueError, match="SHA256"):
        load_detector(path, "0" * 64)


def test_test_split_does_not_change_fit_or_threshold():
    rows = traces()
    before = fit_detector(rows)
    for row in rows:
        if row["split"] == "test":
            row["features"] = dict.fromkeys(FEATURE_NAMES, 10000.)
    assert before == fit_detector(rows)


def test_leaky_groups_and_features_rejected():
    rows = traces(); rows[-1]["group"] = rows[0]["group"]
    with pytest.raises(ValueError, match="disjoint"):
        fit_detector(rows)
    with pytest.raises(ValueError, match="allowlist"):
        feature_vector({**traces()[0]["features"], "target_id": 1})
    rows = traces()
    for r in rows:
        r["variant"] = "known"
    with pytest.raises(ValueError, match="unseen"):
        fit_detector(rows)


def test_parameter_features_are_finite_and_shape_checked():
    assert parameter_features([np.ones(2)], [np.ones(2)])["relative_delta"] == 0
    with pytest.raises(ValueError):
        parameter_features([np.ones(2)], [np.ones(3)])
