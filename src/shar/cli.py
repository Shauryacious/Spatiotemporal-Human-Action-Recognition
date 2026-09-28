import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from shar.config import load_settings
from shar.errors import SharError

app = typer.Typer(no_args_is_help=True, help="Multimodal video action recognition and localization")


@app.command()
def analyze(
    video: Path,
    query: Annotated[str, typer.Option()] = "Find visible human actions",
    output: Annotated[Path, typer.Option()] = Path("outputs/result.json"),
    config: Annotated[Path | None, typer.Option()] = None,
    demo: Annotated[
        bool, typer.Option(help="Scripted synthetic predictions; no real recognition")
    ] = False,
    recognition_only: Annotated[bool, typer.Option()] = False,
):
    from shar.inference.multimodal.mock import MockModel
    from shar.inference.pipeline import Pipeline
    from shar.video.ingest import ingest_path

    overrides = (
        {"enable_temporal": False, "enable_spatial": False, "enable_refinement": False}
        if recognition_only
        else {}
    )
    try:
        settings = load_settings(config, overrides)
        if demo and settings.provider != "mock":
            raise SharError("INVALID_OPTIONS", "Demo mode requires the mock provider")
        if demo:
            settings = settings.model_copy(update={"model": "fixture-demo-v1"})
        asset = ingest_path(video, "cli", settings)
        report = asyncio.run(
            Pipeline(settings, model=MockModel(demo=True) if demo else None).run(asset, query)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2))
        typer.echo(
            json.dumps(
                {
                    "status": report.status,
                    "result": str(output),
                    "model_calls": report.usage.model_calls,
                }
            )
        )
    except (SharError, ValueError, OSError) as exc:
        typer.echo(
            json.dumps(exc.public() if isinstance(exc, SharError) else {"error": str(exc)}),
            err=True,
        )
        raise typer.Exit(1) from exc


@app.command()
def worker(config: Annotated[Path | None, typer.Option()] = None, once: bool = False):
    from shar.workers.runner import Worker

    runner = Worker(load_settings(config))
    asyncio.run(runner.once() if once else runner.run())


@app.command()
def serve(config: Annotated[Path | None, typer.Option()] = None, port: int = 8000):
    import uvicorn

    from shar.api.app import create_app

    settings = load_settings(config)
    uvicorn.run(create_app(settings), host=settings.host, port=port, proxy_headers=False)


@app.command()
def evaluate(
    results: Path,
    annotations: Path,
    output: Path = Path("outputs/metrics.json"),
    recognition_only: bool = False,
    config: Path | None = None,
):
    from shar.evaluation.dataset import Annotation, EvaluationSettings, read_jsonl
    from shar.evaluation.runner import evaluate as score
    from shar.evaluation.runner import read_reports

    spec = EvaluationSettings.load(config)
    value = score(
        read_reports(results),
        read_jsonl(annotations, Annotation),
        recognition_only=recognition_only,
        aliases=spec.aliases,
        thresholds=spec.temporal_thresholds,
        spatial_threshold=spec.spatial_threshold,
        tolerance=spec.sampling_tolerance_sec,
        unknown_score_rank=spec.unknown_score_rank,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2))
    typer.echo(str(output))


@app.command()
def stream(
    source: str,
    config: Path = Path("configs/stream.yaml"),
    duration: float = 60.0,
    output: Path = Path("outputs/stream.jsonl"),
    owner: str = "local",
    query: str = "Describe observable human actions",
):
    from shar.storage.repository import Repository
    from shar.video.stream import admit
    from shar.video.stream import stream as run_stream

    settings = load_settings(config)
    admit(settings)
    if source not in settings.stream_sources:
        raise typer.BadParameter("Source is not in the operator-controlled registry")
    if duration <= 0 or duration > settings.job_timeout_sec:
        raise typer.BadParameter(
            "Duration must be positive and within the configured session deadline"
        )
    repository = Repository(settings)
    session_id = repository.start_stream(owner, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:

        def emit(value):
            value = {"session_id": session_id, **value}
            repository.append_stream_event(session_id, owner, value)
            line = json.dumps(value)
            handle.write(line + "\n")
            handle.flush()
            typer.echo(line)

        emit(
            {
                "type": "session_start",
                "provider": settings.provider,
                "model": settings.model,
                "config_hash": settings.fingerprint(),
            }
        )
        try:
            from shar.storage.artifacts import LocalArtifacts

            artifacts = LocalArtifacts(settings.storage_root)
            asyncio.run(
                run_stream(
                    settings,
                    source,
                    emit,
                    duration,
                    query=query,
                    preview=lambda jpeg: artifacts.put(f"streams/{session_id}/preview.jpg", jpeg),
                )
            )
        except BaseException as exc:
            emit({"type": "session_error", "code": getattr(exc, "code", type(exc).__name__)})
            raise


@app.command()
def cleanup(config: Annotated[Path | None, typer.Option()] = None):
    from shar.workers.cleanup import cleanup as sweep

    typer.echo(json.dumps({"removed_videos": sweep(load_settings(config))}))
