# Spatiotemporal Human Action Recognition

A Python research pipeline for open-vocabulary action recognition, temporal localization,
sparse actor localization, and bounded refinement. Includes a durable FastAPI service,
SQLite worker, browser viewer, independent evaluator, and bounded webcam/RTSP processing.
Gemini is optional; offline tests use a deterministic mock and never require a cloud account.

## Quickstart

Run the following steps in order. Commands below assume macOS or Linux.

1. Install Git, Python 3.12+, [uv](https://docs.astral.sh/uv/), and FFmpeg/FFprobe.
   On macOS with Homebrew:

   ```sh
   brew install git python@3.12 uv ffmpeg
   ```

   On Debian/Ubuntu, install Git and FFmpeg with `sudo apt-get install git ffmpeg`,
   install uv using its linked installation instructions, then run `uv python install 3.12`.

2. Clone the repository and enter it (skip this if you already have a checkout open):

   ```sh
   git clone https://github.com/Shauryacious/Spatiotemporal-Human-Action-Recognition.git
   cd Spatiotemporal-Human-Action-Recognition
   ```

3. Install the locked dependencies, including the Gemini and streaming extras:

   ```sh
   uv sync --locked --all-extras
   ```

4. Generate sample videos and run an offline analysis:

```sh
uv run python scripts/make_fixtures.py
uv run shar analyze data/fixtures/cfr.mp4 --output outputs/result.json
uv run shar analyze data/fixtures/cfr.mp4 --demo --recognition-only --output outputs/demo.json
```

The JSON results are saved to `outputs/result.json` and `outputs/demo.json`.
The default mock returns a valid empty result. `--demo` returns explicitly synthetic
labels to exercise the output path; **neither mode recognizes real actions**.
All times are normalized presentation timestamps, not frame index divided by nominal FPS.
Continue below to run the browser viewer; use the Gemini section for real predictions.

## API and viewer

5. In the terminal where you completed setup, start the worker and leave it running:

```sh
uv run shar worker
```

6. Open a second terminal, change to the same repository directory, and start the API:

```sh
cd /path/to/Spatiotemporal-Human-Action-Recognition
uv run shar serve
```

Replace `/path/to/Spatiotemporal-Human-Action-Recognition` with your checkout's actual path.

7. Open <http://127.0.0.1:8000>. Upload a video (for example, `data/fixtures/cfr.mp4`),
enter an action query, and submit it. Keep both terminals running while using the viewer;
press `Ctrl+C` in each terminal to stop it.

The API persists a queued job immediately; the independent worker publishes its result.
OpenAPI documentation is at `/docs`. `/ready` requires a recent worker heartbeat.
Use the viewer Settings page to set a custom API base URL or the app bearer token for
remote deployments. Local loopback mode does not need a token.

## Gemini

Create a Gemini API key in Google AI Studio, then keep it on the backend as an
environment variable or in a local `.env` file. Do not put provider keys in the browser.

```sh
export GEMINI_API_KEY='your-secret'
export SHAR_PROVIDER=gemini
export SHAR_MODEL='your-accessible-model-id'
export SHAR_RPM='your-account-limit'
export SHAR_TPM='your-account-limit'
export SHAR_RPD='your-account-limit'
uv run shar analyze /path/to/approved-video.mp4 --query 'Find people riding bicycles'
```

If credentials and settings are in `.env`, explicitly load that file in **both** terminals
(the CLI does not load it automatically):

```sh
uv run --env-file .env shar worker
uv run --env-file .env shar serve
```

The viewer enables actor localisation by default. Results show action confidence,
start/end timestamps, duration, a seekable timeline, and labeled observed boxes.
Use “Show first observed box” to inspect spatial evidence. Boxes are sparse observations,
not continuous tracking; confidence scores are uncalibrated model estimates.
A previous mock result must be rerun with Gemini to obtain real predictions.

Replace limit placeholders with integers. There is no automatic model selection or
fallback. Credentials are never stored in result files. Gemini receives sampled RGB
JPEGs and the action query; only use footage approved for that transfer. See
[operations](docs/operations.md) for authentication, quotas, retention, and recovery.

## Evaluation and streaming

```sh
uv run python scripts/run_experiment.py data/manifest.jsonl data/annotations.jsonl --output outputs/experiment
uv run shar evaluate outputs/experiment/B2/0/results data/annotations.jsonl
uv run shar stream webcam --config configs/stream.yaml --duration 60
```

Streaming requires the `stream` extra and an operator-registered source. Live Gemini
admission checks worst-case request/token/day demand including retries; a nominal
four-second stride is not a promise of free or real-time operation.

## Verification

```sh
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests migrations
uv run mypy src/shar
uv build
```

See [architecture](docs/architecture.md), [API](docs/api.md),
[research protocol](docs/research_protocol.md), and [experiment instructions](experiments/README.md).
The original [implementation plan](IMPLEMENTATION_PLAN.md) records the target design;
implementation status and external validation limits are documented in [verification](docs/verification.md).

The stream command emits a session ID. Paste it into the viewer's stream panel to see
persisted revisions and the latest preview; inference delays do not block preview capture.

Optional verification with installed Chrome:

```sh
uv run --with playwright python scripts/smoke_viewer.py
uv run python scripts/profile_pipeline.py --duration 10
```
# Spatiotemporal-Human-Action-Recognition
