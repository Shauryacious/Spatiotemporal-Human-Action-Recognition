# Analysis failure investigation

The saved run `6bcfd1f7ec7643ca82b3e149fefcc481` failed on its first temporal
inference request. The worker log records a Gemini `ServerError`, HTTP 503, with no
provider retry delay. Upload and decoding had completed; no model result was returned.

The run snapshot and local `.env` both specified `max_calls=1` and
`max_job_attempts=1`. The pipeline limits request attempts to the remaining model-call
budget, so the intended three attempts became one. Raising the timeout from 60 to
180 seconds could not recover an explicit HTTP 503 response. The same one-call limit
also left no capacity for the requested refinement and actor localization stages.
Earlier runs separately encountered daily quota exhaustion; that requires a quota
reset or account configuration change, rather than transient-error retries.

## Changes

- Local `.env` now permits at most 12 total model calls per run, with account quota
  limits unchanged. Each request gets at most three attempts. Worker restarts do not
  multiply an exhausted request's retries.
- Retry backoff starts at two seconds with exponential growth and jitter. Provider
  `Retry-After` delays remain a lower bound, and waits respect cancellation and the
  overall job deadline.
- Local capacity waits no longer consume provider attempt slots. A one-call run can
  wait for capacity and still make its one permitted request.
- Shared cooldowns preserve the retryable category of temporary overload, timeout,
  and minute-rate errors. Daily and zero quota errors remain terminal.
- HTTPX and optional aiohttp connection failures are classified as recoverable
  transport errors rather than generic permanent provider failures.
- Terminal transient errors explain the exhausted retry count or model-call limit.
  Retry activity records the error category and scheduled retry time in run history.
- API and worker were restarted to load the code and environment changes. Existing
  history is preserved; existing run settings are immutable snapshots.

## Verification

49 focused tests passed across retry recovery, recognition, workers, Gemini SDK
handling, quota limits, caching, and settings. The SDK recovery test sends two HTTP
503 responses followed by a valid response through a fake HTTP transport and verifies
that the same run succeeds on exactly three calls and one worker attempt. Additional
regressions cover repeated capacity waits, cross-process cooldowns, cancellation,
deadlines, network errors, and an explicitly configured one-call budget.

## Live findings

Run `2a8de100fbea4fda8670e335ee79ac90` retried the original model correctly but
received HTTP 503 twice followed by HTTP 504. This confirms a continuing provider
availability problem, separate from the original one-call configuration bug.

The account's model catalog exposed `gemini-3.5-flash-lite`. Run
`c526861374614ab1a67401785b8009b0` used that explicit model with the original video,
query, actor-localization option, and boundary-refinement option. It recognized waving
and refined the interval to approximately 1.268–3.003 seconds. It used eight calls and
finished partially because its spatial response failed validation.

Replaying the saved rejected spatial response against its saved frame manifest
identified the exact error: `boundary outside request`. Its actor tracks alone pass
validation. The shared schema and spatial prompt's temporal example invited the model
to copy a temporal boundary whose uncertainty bracket starts before the spatial
sampling interval. The spatial task now requests only `tracks` in both its prompt and
provider schema; unused temporal schema definitions and examples are removed. Local
frame, timestamp, box, and evidence validation remain strict.

Local `.env` now selects the explicit `gemini-3.5-flash-lite` model instead of the
unavailable `gemini-flash-latest` alias. This changes the inference model; it is a
latency/cost-oriented model, not a claim of identical accuracy across all videos.

No provider outage can be made impossible by application code.

Provider reference: [Gemini troubleshooting](https://ai.google.dev/gemini-api/docs/troubleshooting).

Model reference: [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite).

## Final result

After the spatial contract fix and service restart, run
`b2ec4fe3fa504f13bf98ed50d3130b05` completed with `status=succeeded` and no run
warnings. It detected waving from 1.2679333333 to 3.003 seconds, with one actor track
and seven observed boxes. Spatial coverage is sparse (`spatial_status=partial`), as
expected for sampled observations; no boxes are interpolated. Six cached responses
were reused, with only one new model call for spatial localization.

The additional SDK/schema/recognition regression suite passed all 29 tests, including
the restricted spatial schema through the real SDK's HTTP serialization. Lint passed.
The services now load `gemini-3.5-flash-lite` as the local default model.

## Follow-up: explicit retry consent and refinement cost

Run `60641f224af04dd2b21f1dd54e8f9540` was cancelled by the user after five model
calls. The saved activity records one temporal call, three boundary-search calls,
and a call in a second refinement pass. It records no provider failure. The repeated
`attempt 1/3` labels referred to separate requests, making the cost and stage state
unclear. One checkbox had enabled two iterative procedures.

The interactive policy now supersedes the earlier automatic retry behavior:

- One refinement checkbox enables one pass with at most one call per action.
- The default new-call budget is three; retry approval defaults to one new call.
- Any failed request stops further model calls, preserving successful output and
  explaining the cause. Correction requests also require approval.
- Analyze and History offer explicit retry approval or keeping results without calls.
- A linked retry replays durable, integrity-checked successful responses for free.
  Missing/corrupted saved output blocks regeneration. Duplicate approvals and worker
  lease expiry cannot silently create additional paid attempts.
- Cancellation is described as cancellation, not a refinement failure.

This follow-up was tested with mock providers, a fake SDK HTTP transport, and an offline
browser. No live Gemini calls were made to validate the changes.
