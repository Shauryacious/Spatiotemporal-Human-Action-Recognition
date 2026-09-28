import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from shar.api.auth import owner
from shar.api.routes import router
from shar.config import load_settings
from shar.errors import SharError
from shar.schemas import uid
from shar.storage.artifacts import LocalArtifacts
from shar.storage.repository import Repository


class UploadLimit:
    """Reject oversized ASGI bodies before multipart spooling can exhaust disk."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.limit if scope.get("path") == "/videos" else 65536
        if scope.get("path") == "/videos" and scope.get("method") == "POST":
            try:
                owner(Request(scope))
            except SharError as error:
                return await JSONResponse(error.public(uid()), status_code=error.status)(
                    scope, receive, send
                )
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                oversized = int(headers[b"content-length"]) > limit
            except ValueError:
                oversized = True
            if oversized:
                size_error = SharError(
                    "VIDEO_TOO_LARGE", "Request body exceeds size limit", status=413
                )
                return await JSONResponse(size_error.public(uid()), status_code=413)(
                    scope, receive, send
                )
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                raise SharError("VIDEO_TOO_LARGE", "Request body exceeds upload limit", status=413)
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None):
    settings = settings or load_settings()
    if settings.auth_mode == "token" and not json.loads(os.environ.get("SHAR_AUTH_TOKENS", "{}")):
        raise ValueError("Token authentication requires SHAR_AUTH_TOKENS")
    app = FastAPI(title="Spatiotemporal Human Action Recognition", version="0.1.0")
    app.state.settings = settings
    app.state.repo = Repository(settings)
    app.state.store = LocalArtifacts(settings.storage_root)
    app.add_middleware(UploadLimit, limit=settings.max_upload_bytes + 65536)

    @app.exception_handler(SharError)
    async def errors(request: Request, error: SharError):
        return JSONResponse(error.public(uid()), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse(
            SharError("INVALID_REQUEST", "Request parameters failed validation").public(uid()),
            status_code=422,
        )

    @app.exception_handler(SQLAlchemyError)
    async def unavailable(request: Request, error: SQLAlchemyError):
        return JSONResponse(
            SharError(
                "PERSISTENCE_UNAVAILABLE", "Persistence is temporarily unavailable", True, 503
            ).public(uid()),
            status_code=503,
        )

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def viewer():
        return FileResponse(
            Path(__file__).parents[1] / "web" / "viewer.html", media_type="text/html"
        )

    app.include_router(router)
    return app
