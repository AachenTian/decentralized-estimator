from private_mb_sac.core.config import expected_interaction_counts


def test_smoke_interaction_count():
    per_owner, system = expected_interaction_counts(
        rounds=1,
        num_owners=3,
        num_envs_per_owner=2,
        rollout_length=25,
    )
    assert per_owner == 50
    assert system == 150


def test_formal_interaction_count():
    per_owner, system = expected_interaction_counts(
        rounds=200,
        num_owners=3,
        num_envs_per_owner=12,
        rollout_length=25,
    )
    assert per_owner == 60_000
    assert system == 180_000
