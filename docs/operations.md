# Operations

Use one host and one worker initially. Keep API and worker storage paths/config identical.
Do not put SQLite WAL on network storage. Run `uv run alembic upgrade head` when provisioning
an empty database; development startup also creates the initial schema idempotently.

Set explicit Gemini model and actual account RPM/TPM/RPD; no historical quota constants
are assumed. The pinned adapter uses the official `google-genai` asynchronous methods.
Its request construction follows [SDK documentation](https://googleapis.github.io/python-genai/).
Prices are optional recorded per-million-token settings. Unknown usage/cost stays null.
A cost cap requires both prices. Conservative reservations include output capacity; the
limiter reconciles measured usage when returned. Quotas are shared only within this storage
root; separate applications/projects need an external common limiter.

Workers heartbeat every lease/3 seconds. A crash leaves a lease that a restarted worker
can reclaim after expiry. Three job attempts are the default dead-letter limit. Durable
call counts survive reclamation; cancellation invalidates the lease immediately. A lost
lease prevents final publication even if an in-flight external request completes.

Run `uv run shar cleanup` periodically. It expires completed videos/results, old cache entries,
and orphan result publications. It does not delete assets with active jobs. Back up the
SQLite database and managed artifacts together while the worker is stopped. Do not back up
or publish provider keys. Runtime files, media, outputs, and `.env` are ignored by Git.

Remote URLs are disabled. Camera sources are operator-configured in YAML, never supplied
by arbitrary API clients. Local-mode bypass accepts only loopback; forwarded client headers
are not trusted by the CLI server. Auth tokens belong in environment variables. The viewer
can save its entered API base URL and bearer token in browser local storage and renders
generated labels via textContent.

Container deployment uses separate API and worker services and a shared local volume.
Configure tokens before starting compose. Container execution and real camera/provider
soak tests must be verified on the deployment host; see verification notes.

On macOS with an iCloud-managed Desktop, the filesystem can evict environment files or
mark `.pth` files hidden. Python then skips the editable-install path, producing
`ModuleNotFoundError: shar.cli`. Keep virtual environments on a local, non-synced disk
(`UV_PROJECT_ENVIRONMENT=/absolute/local/path`) or use `uv sync --locked --all-extras --no-editable`.
This is separate from application import/package correctness. The release wheel includes
all Python modules, prompt templates, and viewer assets.

Uploads are authenticated before multipart parsing and bounded before spooling. Probe and
PTS indexing run in a separate process with a wall-clock timeout and POSIX CPU limits;
Linux also applies a 2 GiB address-space ceiling. Source media and exact PTS indices are
retained together. Capture preview is independent of inference and overwrites a single
managed JPEG. Stream events persist in SQLite and JSONL; retention removes expired sessions
and previews. A restarted stream gets a new session ID; reconnects within it get new epochs.
