from shar.telemetry import redact


def test_redaction():
    assert redact({"api_key": "secret", "nested": {"authorization": "bearer", "calls": 2}}) == {
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]", "calls": 2},
    }
