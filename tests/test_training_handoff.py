from training_preparation.prepare_handoff import _split_groups


def test_grouped_split_is_reproducible_disjoint_and_keeps_duplicates_together():
    groups = {f"trajectory-{episode}": [episode] for episode in range(30)}
    groups["trajectory-0"] = [0, 30]

    first = _split_groups(groups, seed=17, n_episodes=31,
                          ratios={"train": 0.7, "validation": 0.15, "test": 0.15})
    second = _split_groups(groups, seed=17, n_episodes=31,
                           ratios={"train": 0.7, "validation": 0.15, "test": 0.15})

    assert first == second
    assert first is not None
    splits = [set(first[name]) for name in ("train", "validation", "test")]
    assert not any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3))
    assert any({0, 30}.issubset(values) for values in splits)
    assert set.union(*splits) == set(range(31))


def test_split_is_blocked_when_fewer_than_three_numeric_trajectory_groups():
    groups = {"same-trajectory": list(range(10))}
    assert _split_groups(groups, seed=17, n_episodes=10,
                         ratios={"train": 0.7, "validation": 0.15, "test": 0.15}) is None
