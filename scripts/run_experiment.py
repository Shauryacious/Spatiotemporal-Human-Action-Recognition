import argparse
import asyncio
import json
from pathlib import Path

from shar.config import load_settings
from shar.evaluation.dataset import EvaluationSettings, load_dataset
from shar.evaluation.runner import evaluate, freeze
from shar.inference.pipeline import Pipeline
from shar.video.ingest import ingest_path


async def run(manifest, annotations, output, config=None, repeats=1, evaluation_config=None):
    records, truth = load_dataset(manifest, annotations)
    if len({r.split for r in records}) != 1:
        raise ValueError("An experiment must select exactly one split")
    base = load_settings(config)
    spec = EvaluationSettings.load(evaluation_config)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    summaries = {}
    for baseline in ("B1", "B2", "B3", "B4"):
        for repeat in range(repeats):
            destination = output / baseline / str(repeat)
            settings = base.model_copy(
                update={
                    "enable_temporal": baseline != "B1",
                    "enable_spatial": baseline in {"B3", "B4"},
                    "enable_refinement": baseline == "B4",
                }
            )
            freeze(
                destination,
                settings,
                manifest,
                annotations,
                {
                    "baseline": baseline,
                    "repeat": repeat,
                    "evaluation": spec.model_dump(mode="json"),
                },
            )
            reports, failures = [], []
            results = destination / "results"
            results.mkdir(exist_ok=True)
            # Inference receives only media and query. Annotation objects stay in evaluator.
            for record in records:
                try:
                    asset = ingest_path(manifest.parent / record.path, "experiment", settings)
                    asset.id = record.video_id
                    report = await Pipeline(settings).run(asset, "Find visible human actions")
                    reports.append(report)
                    (results / f"{record.video_id}.json").write_text(
                        report.model_dump_json(indent=2)
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "video_id": record.video_id,
                            "code": getattr(exc, "code", type(exc).__name__),
                        }
                    )
            metrics = evaluate(
                reports,
                truth,
                recognition_only=baseline == "B1",
                aliases=spec.aliases,
                thresholds=spec.temporal_thresholds,
                spatial_threshold=spec.spatial_threshold,
                tolerance=spec.sampling_tolerance_sec,
                unknown_score_rank=spec.unknown_score_rank,
            )
            metrics["failures"] = failures
            (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
            summaries[f"{baseline}/{repeat}"] = metrics
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--evaluation-config", type=Path)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.manifest,
            args.annotations,
            args.output,
            args.config,
            args.repeats,
            args.evaluation_config,
        )
    )
