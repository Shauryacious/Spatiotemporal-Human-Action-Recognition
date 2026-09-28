import hashlib
import json

from scripts.run_experiment import run


async def test_four_baselines_freeze_and_score(media, tmp_path, monkeypatch):
    monkeypatch.setenv("SHAR_STORAGE_ROOT", str(tmp_path / "runtime"))
    video = media / "cfr.mp4"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "path": str(video),
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "duration_sec": 4.0,
                "split": "validation",
                "group_id": "g",
                "license": "synthetic",
            }
        )
        + "\n"
    )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text('{"video_id":"v1","actions":[],"labels":[],"tracks":[]}\n')
    summaries = await run(manifest, annotations, tmp_path / "experiment")
    assert len(summaries) == 4
    assert summaries["B1/0"]["temporal"] is None
    assert not summaries["B4/0"]["failures"]
    frozen = json.loads((tmp_path / "experiment/B4/0/manifest.json").read_text())
    assert frozen["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
