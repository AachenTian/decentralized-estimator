from private_mb_sac.rollout.model_rollout import model_rollout_horizon


def test_rollout_horizon_schedule():
    kwargs=dict(horizon_min=1,horizon_max=6,schedule_start_round=15,schedule_end_round=100)
    assert model_rollout_horizon(1,**kwargs)==1
    assert model_rollout_horizon(15,**kwargs)==1
    assert model_rollout_horizon(100,**kwargs)==6
    assert model_rollout_horizon(150,**kwargs)==6
