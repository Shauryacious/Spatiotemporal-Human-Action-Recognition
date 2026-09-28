import hmac
import json
import os

from fastapi import Request

from shar.errors import SharError


def owner(request: Request) -> str:
    settings = request.app.state.settings
    if settings.auth_mode == "local":
        if (
            settings.host not in {"127.0.0.1", "::1", "localhost"}
            or not request.client
            or request.client.host not in {"127.0.0.1", "::1", "localhost", "testclient"}
        ):
            raise SharError("UNAUTHORIZED", "Local mode accepts loopback clients only", status=401)
        return "local"
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        raise SharError("UNAUTHORIZED", "Bearer token required", status=401)
    supplied = authorization[7:]
    tokens = json.loads(os.environ.get("SHAR_AUTH_TOKENS", "{}"))
    for token, identity in tokens.items():
        if hmac.compare_digest(supplied, token):
            return identity
    raise SharError("UNAUTHORIZED", "Invalid bearer token", status=401)
