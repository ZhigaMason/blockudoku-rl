import pytest

from blockudoku.config import CONFIG_DIR, load_config


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_file_loads(path):
    cfg = load_config(path)
    assert cfg.v_max > cfg.v_min and cfg.n_step >= 1


def test_extends_and_overrides():
    cfg = load_config("smoke", ["net.dim=48", "gamma=0.5", "learning_rate=3e-4"])
    full = load_config("full")
    assert cfg.net.dim == 48 and cfg.gamma == 0.5 and cfg.learning_rate == 3e-4
    assert cfg.net.rope_base == full.net.rope_base and cfg.gamma != full.gamma  # via `extends`


def test_roundtrip(tmp_path):
    cfg = load_config("small")
    cfg.save(tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_rejects_unknown_and_missing_keys(tmp_path):
    with pytest.raises(ValueError, match="unknown keys.*typo"):
        load_config("smoke", ["typo=1"])
    (tmp_path / "partial.yaml").write_text("seed: 1\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_config(tmp_path / "partial.yaml")


def test_wandb_section_is_validated():
    assert load_config("full").wandb.enabled and not load_config("smoke").wandb.enabled
    with pytest.raises(ValueError, match="wandb.enabled must be true/false"):
        load_config("smoke", ["wandb.enabled=maybe"])
    with pytest.raises(ValueError, match="wandb.mode must be one of"):
        load_config("smoke", ["wandb.mode=cloud"])
