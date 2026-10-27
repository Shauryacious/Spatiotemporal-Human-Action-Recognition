import pytest

from shar.config import Settings, load_settings


def test_config_precedence(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("coarse_fps: 2\nmodel: yaml-model\n")
    monkeypatch.setenv("SHAR_MODEL", "env-model")
    assert load_settings(config).model == "env-model"
    assert load_settings(config, {"coarse_fps": 3}).coarse_fps == 3
    assert "storage_root" not in load_settings(config).public()


def test_env_call_budget_is_loaded(monkeypatch):
    monkeypatch.setenv("SHAR_MAX_CALLS", "1")
    monkeypatch.setenv("SHAR_MAX_JOB_ATTEMPTS", "1")
    monkeypatch.setenv("SHAR_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("SHAR_THINKING_LEVEL", "low")
    monkeypatch.setenv("SHAR_CALL_TIMEOUT_SEC", "180")
    monkeypatch.setenv("SHAR_PROVIDER_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("SHAR_RETRY_BASE_SEC", "3")
    monkeypatch.setenv("SHAR_RETRY_MAX_SEC", "45")
    settings = load_settings()
    assert settings.max_calls == settings.max_job_attempts == 1
    assert settings.output_tokens == 8192
    assert settings.thinking_level == "low"
    assert settings.call_timeout_sec == 180
    assert settings.provider_max_attempts == 4
    assert settings.retry_base_sec == 3
    assert settings.retry_max_sec == 45


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_sec": 5, "overlap_sec": 5},
        {"host": "0.0.0.0"},
        {"provider": "gemini"},
        {"enable_temporal": False},
        {"retry_base_sec": 5, "retry_max_sec": 1},
        {"provider_max_attempts": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
