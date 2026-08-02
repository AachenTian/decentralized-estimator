import yaml


def test_required_animation_rounds_are_configured():
    config = yaml.safe_load(
        open("configs/default.yaml", encoding="utf-8")
    )
    assert config["environment_rng"]["forced_seed"] == 40
    assert config["evaluation"]["environment_seed"] == 40
    assert config["animation"]["rounds"] == [
        0, 10, 30, 60, 90, 120, 150, 180, 190, 200
    ]
