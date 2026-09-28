import json
import logging
import re
import time
from contextlib import contextmanager

SECRET = re.compile(
    r"api.?key|(^|_)(token|secret|authorization|password|prompt|query|jpeg|uri|url|source)(_|$)",
    re.IGNORECASE,
)


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET.search(k) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def event(name: str, **fields):
    logging.getLogger("shar").info(json.dumps({"event": name, **redact(fields)}, default=str))


@contextmanager
def timer(name: str, **fields):
    started = time.monotonic()
    try:
        yield
    finally:
        event(name, elapsed_sec=time.monotonic() - started, **fields)
