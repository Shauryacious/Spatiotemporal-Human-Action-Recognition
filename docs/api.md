# API guide

Local development binds loopback. For shared deployments set `SHAR_AUTH_MODE=token`,
`SHAR_AUTH_TOKENS='{"long-random-token":"owner-id"}'`, and `SHAR_HOST=0.0.0.0`.
Use `Authorization: Bearer ...` on every data route. TLS belongs at the reverse proxy.

```sh
curl -F file=@data/fixtures/cfr.mp4 http://127.0.0.1:8000/videos
curl -H 'Content-Type: application/json' -H 'Idempotency-Key: example-1' \
  -d '{"video_id":"REPLACE","prompt":"Find walking","enable_refinement":false}' \
  http://127.0.0.1:8000/localize
curl http://127.0.0.1:8000/jobs/REPLACE
curl http://127.0.0.1:8000/results/REPLACE
```

| Route | Result |
|---|---|
| `POST /videos` | Multipart `file`, 201 video ID and metadata |
| `POST /jobs` | Canonical asynchronous submission, 202 and Location |
| `POST /analyze` | Recognition by default; explicit temporal options supported |
| `POST /localize` | Temporal localization, optional spatial/refinement |
| `POST /track` | Requires interval or prediction ID + source job ID |
| `GET /jobs/{id}` | Owner-scoped progress, attempts, calls, safe error code |
| `GET /results/{id}` | Immutable report; 409 if no completed result |
| `POST /jobs/{id}/cancel` | Idempotent cancellation, 202 |
| `GET /videos/{id}/content` | Owner-scoped video with HTTP range support |
| `GET /health` | Process liveness |
| `GET /ready` | Database and worker heartbeat readiness |

Submission accepts `prompt`, `video_id`, feature booleans, `interval: [start,end]`, and
`config_profile: "research-default"`. Arbitrary remote URL ingestion is disabled.
Request callers cannot override server paths, credentials, model, or quotas. Idempotency
keys are owner-scoped; reuse with a different request returns 409. Missing and foreign
resources both return 404. Errors use a code/message/request ID/retryable envelope.

Operator-started streams emit a session ID. `GET /streams/{id}` returns its state;
`GET /streams/{id}/events?after=SEQUENCE` returns up to 200 later persisted revisions;
`GET /streams/{id}/preview` returns the latest compressed observation. All three are
owner-scoped. The viewer's stream panel polls these routes. Use `shar stream --owner OWNER`
to match an authenticated owner's ID. Stream startup is operator-controlled through the CLI,
so API clients cannot register arbitrary camera URLs.
