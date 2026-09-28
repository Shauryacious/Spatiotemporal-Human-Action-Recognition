# Reproducible experiments

A manifest is JSONL with one record per video:

```json
{"video_id":"v1","path":"videos/v1.mp4","sha256":"actual-content-hash","duration_sec":4.0,"split":"validation","group_id":"session1","license":"consented pilot"}
```

Paths are relative to the manifest. Annotations are separate JSONL:

```json
{"video_id":"v1","labels":["walking"],"actions":[{"id":"gt1","label":"walking","start_sec":1.0,"end_sec":3.0,"actor_ids":[]}],"tracks":[]}
```

Use the canonical `ActorTrack`/`SpatialDetection` schema for observed boxes; actor IDs
in an action must reference those tracks. Every video needs an annotation record, including
`{"video_id":"empty","labels":[],"actions":[],"tracks":[]}` for a no-action scene.

`uv run python scripts/run_experiment.py MANIFEST ANNOTATIONS --repeats 3` runs B1–B4,
freezes provenance, writes immutable per-video reports, records failures, and evaluates
independently. Results live under `outputs/experiment/{baseline}/{repeat}`. No annotation
is passed to the inference pipeline. Do not compare warm-cache latency with cold runs.

Use `--evaluation-config configs/evaluation.yaml` on the experiment runner, or
`shar evaluate ... --config configs/evaluation.yaml`, to freeze an alias mapping,
diagnostic tIoU thresholds, joint spatial threshold, observation tolerance, and null-score
ranking policy independently of inference NMS. Configuration files must not contain test-tuned
aliases or thresholds. Native-video ablations use `input_strategy: native_video` with Gemini;
the adapter uploads silent clips and converts relative times exactly once.
