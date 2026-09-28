# Implementation and verification record

Verified locally on 28 September 2026 with Python 3.12.2, uv 0.10.8, FFmpeg, and the
committed dependency lock. Application version: 0.1.0.

## Delivered

- Strict media, evidence, prediction, boundary, track, result, and job contracts; validated
  YAML/environment configuration, safe errors, JSON telemetry, and versioned prompts.
- Managed bounded ingestion; isolated probe/index process; CFR/VFR presentation timestamps,
  disk-backed PTS lookup, rotation, RGB transforms, uniform and optional motion sampling.
- Provider-neutral mock/Gemini adapters, timestamped images and silent native-video clips,
  provider schema translation, owned remote-file cleanup, shared quota reservations,
  bounded retries, durable call/spend admission, scoped validated caching, and model-drift checks.
- Recognition-only output, temporal proposals and local boundary evidence, conservative
  continuity stitching/NMS, sparse gated actor association, and bounded refinement with rollback.
- SQLite/WAL durable jobs, migrations, owner-scoped idempotency, leases/fencing, cancellation,
  atomic result publication, retention cleanup, asynchronous API, and HTML video viewer.
- Independent recognition/temporal/spatial/joint evaluation, split/hash checks, frozen
  evaluation aliases/thresholds, and B1–B4 experiment runner with per-run artifacts and failures.
- Registered-source capture, bounded ring and pending window, dropped/failed coverage,
  reconnect epochs, event revisions, persistent event polling, and independent preview.
- Locked packaging, CI definition, container/compose recipes, operations/API/research docs.

## Checks executed

| Check | Outcome |
|---|---|
| Unit/integration/end-to-end suite | 67 tests passed |
| Ruff lint and formatting | Passed |
| Mypy | Passed, 49 source files |
| Dependency lock validation | Passed |
| Source distribution and wheel build | Passed |
| Wheel installed into a separate environment | Mock analysis and synthetic recognition CLI passed |
| Actual Google SDK serialization with fake HTTP transport | Passed; no paid/network inference |
| Real Chrome viewer smoke | Upload → job → worker → result/download passed; stream polling/preview passed; no JavaScript errors |
| Compose configuration validation | Passed |
| Synthetic slow-provider stream soak | Bounded 10-frame ring; dropped windows explicitly reported |

The 10-second synthetic soak processed 38 windows, reported 123 dropped windows, and ended
with 10 retained frames. Observed frame-age p50/p95 were approximately 0.336/0.393 seconds;
process peak RSS was about 90.0 MB including interpreter/imports. This is a short local
stress check, **not** a real camera/cloud latency or long-running memory benchmark.
Reproduce with `uv run python scripts/profile_pipeline.py --duration 10`.

The test run emits a Starlette warning that its HTTPX-based TestClient is deprecated.
It does not fail the tests; all dependency versions are locked.

## External validation still required

- No live Gemini inference was run: an accessible model, actual account quotas, credentials,
  and approved footage are needed. Mock outputs establish engineering behavior only.
- No real webcam/RTSP or long-duration deployed soak was run. Capture recovery and backpressure
  were exercised with synthetic sources; validate the actual camera/backend on its host.
- Docker runtime/build validation was unavailable because the local Docker daemon is stopped.
  The compose file was validated; the container recipes must still be exercised with a daemon.
- No licensed pilot/benchmark videos or reviewed action/box annotations were supplied.
  Consequently there are no scientific accuracy claims, real B1–B4 comparisons, confidence
  intervals, or claims that iterative refinement improves localization. The runner and
  evaluator are ready for those experiments; the research acceptance gate remains open.

Arbitrary remote video URLs remain disabled by design. Streams are started by operators
through the CLI; the API exposes owner-scoped observation and revision polling. SQLite
assumes one host with local storage. Actor tracks are sparse hypotheses, not reliable
person identification or dense tracking.

The workspace's generated virtual environment was relocated to a local cache directory
and `.venv` links to it, because iCloud marked editable-install path files hidden. This
machine-specific environment link is ignored by Git. A clean checkout creates its own
environment with the README commands.
