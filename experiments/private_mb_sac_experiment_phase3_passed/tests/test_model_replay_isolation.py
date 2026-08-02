from types import SimpleNamespace
from private_mb_sac.training.model_rollout import initialize_owner_model_runtimes


def test_model_replays_are_private():
    config=SimpleNamespace(experiment=SimpleNamespace(num_agents=3),replay=SimpleNamespace(model_capacity=32),actor=SimpleNamespace(observation_dim=18,action_dim=2),critic=SimpleNamespace(state_dim=18))
    runtimes=initialize_owner_model_runtimes(config)
    assert len({id(x.model_replay) for x in runtimes})==3
