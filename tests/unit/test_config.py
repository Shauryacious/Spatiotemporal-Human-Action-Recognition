import pytest

from shar.config import Settings, load_settings


def test_config_precedence(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("coarse_fps: 2\nmodel: yaml-model\n")
    monkeypatch.setenv("SHAR_MODEL", "env-model")
    assert load_settings(config).model == "env-model"
    assert load_settings(config, {"coarse_fps": 3}).coarse_fps == 3
    assert "storage_root" not in load_settings(config).public()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_sec": 5, "overlap_sec": 5},
        {"host": "0.0.0.0"},
        {"provider": "gemini"},
        {"enable_temporal": False},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
