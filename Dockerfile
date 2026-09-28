FROM python:3.12.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.10.8
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY prompts ./prompts
RUN uv sync --locked --no-dev --extra gemini
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./
RUN useradd --create-home --uid 10001 shar && mkdir -p /app/runtime && chown -R shar:shar /app/runtime
USER shar
ENV PATH="/app/.venv/bin:$PATH" SHAR_STORAGE_ROOT=/app/runtime
EXPOSE 8000
HEALTHCHECK --interval=30s CMD curl --fail http://127.0.0.1:8000/health || exit 1
CMD ["shar", "serve"]
