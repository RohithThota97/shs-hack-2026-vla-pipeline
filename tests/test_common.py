from hackpipe.common.ids import demo_id, fingerprint, next_version
from hackpipe.common.stats import percentile, wilson_interval


def test_wilson_matches_known_value():
    lo, hi = wilson_interval(17, 20)
    assert round(lo, 2) == 0.64 and round(hi, 2) == 0.95


def test_wilson_edges():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo > 0.8


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_ids_deterministic():
    assert demo_id("src", 3) == demo_id("src", 3)
    assert demo_id("src", 3) != demo_id("src", 4)
    assert fingerprint(["b", "a"], {"x": 1}) == fingerprint(["a", "b"], {"x": 1})
    assert fingerprint(["a"], {"x": 1}) != fingerprint(["a"], {"x": 2})


def test_next_version():
    assert next_version([], "ds") == "ds-v1"
    assert next_version(["ds-v1", "ds-v3", "m-v9"], "ds") == "ds-v4"
