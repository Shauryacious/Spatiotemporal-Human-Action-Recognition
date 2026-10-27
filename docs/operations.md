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

Provider calls retry at most three times; exhausting those retries does not restart the
whole job. Gemini daily quota exhaustion or a zero quota stops immediately. Short-term
rate limits honor the provider's `RetryInfo` / `Retry-After` delay; a delay longer than
the remaining job deadline fails immediately with a retry time. Job activity retains a
safe explanation and retry time, without storing raw provider errors or credentials.
The viewer shows these details for failed analyses.

Provider cooldowns persist in `quotas.sqlite`, scoped by model and a credential digest
(never the key itself). The viewer checks saved pauses before uploading, and workers
enforce them across jobs without incrementing model-call counts. If you increase your
provider quota, use Settings → "I updated my provider quota" to permit a fresh attempt.
Changing the backend credential naturally selects its own cooldown state.

In the viewer, disabling "Refine boundaries" disables both boundary search and iterative
refinement. Disable both refinement and actor localization for fewer model calls.

For a strict one-call-per-job profile, set `SHAR_MAX_CALLS=1` and
`SHAR_MAX_JOB_ATTEMPTS=1` in `.env`, restart both services, and leave the viewer's
optional extra passes disabled. The viewer defaults to these extra passes being off.
This profile analyzes at most one model window per job. Provider errors are retained
as the original error rather than hidden behind a subsequent call-budget failure.
Rejected model output is saved privately as `runtime/cache/*.invalid.json` for offline
diagnosis. No additional model request is required to inspect it.

`SHAR_CALL_TIMEOUT_SEC` controls the full model-call timeout. `SHAR_OUTPUT_TOKENS`
controls response capacity; `SHAR_THINKING_LEVEL` optionally selects a thinking level
supported by the configured Gemini model. The SDK's internal retries remain disabled.

Local `SHAR_RPM`, `SHAR_TPM`, and `SHAR_RPD` values are guardrails, not purchased quota.
Set them to the actual limits in Google AI Studio. A larger local value cannot override
Google's project/model limits. After changing Python code, restart both `shar serve`
and `shar worker`; these commands do not reload Python modules automatically.

See [free API setup](free-api-setup.md) for key creation, free-tier limits, and provider
compatibility. Keys from other services cannot be used as `GEMINI_API_KEY`.

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

### Local daily request exhaustion (`DAILY_QUOTA`)

`SHAR_RPD` limits durable reservations per model/credential over a rolling 24-hour
window. It is independent of provider quota and includes failed or uncertain calls.
Cached responses consume no reservation. A run can therefore report zero new calls
and still stop when its next uncached stage needs capacity.

Daily exhaustion now reports when enough reservations expire, including after a cap
is lowered. Analyze/History recompute this time for saved failures and allow an
explicit approved retry once capacity returns. Completed responses are reused.
There is no automatic daily-quota retry. Restart the API and worker to load changes.
Do not delete quota records or increase `SHAR_RPD` merely to bypass the stop; change
the cap only after verifying the provider allowance for the configured account/model.
