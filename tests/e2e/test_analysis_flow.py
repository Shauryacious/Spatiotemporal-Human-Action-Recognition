from fastapi.testclient import TestClient

from shar.api.app import create_app
from shar.workers.runner import Worker


async def test_upload_to_result(settings, media):
    client = TestClient(create_app(settings))
    with (media / "cfr.mp4").open("rb") as f:
        response = client.post("/videos", files={"file": ("video.mp4", f, "video/mp4")})
    assert response.status_code == 201, response.text
    body = {
        "video_id": response.json()["video_id"],
        "prompt": "find walking",
        "enable_refinement": False,
    }
    response = client.post("/localize", json=body, headers={"Idempotency-Key": "abc"})
    assert response.status_code == 202, response.text
    job = response.json()["job_id"]
    assert (
        client.post("/localize", json=body, headers={"Idempotency-Key": "abc"}).json()["job_id"]
        == job
    )
    assert client.get("/results/" + job).status_code == 409
    assert await Worker(settings).once()
    result = client.get("/results/" + job)
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "succeeded"
    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    assert (
        client.get(
            "/videos/" + body["video_id"] + "/content", headers={"Range": "bytes=0-99"}
        ).status_code
        == 206
    )


async def test_track_uses_supplied_region(settings, asset):
    from shar.inference.multimodal.mock import MockModel
    from shar.schemas import ActorTrack, BBox, ModelPayload, SpatialDetection

    seen = []

    def response(request):
        seen.append(request.task)
        f = request.frames[0]
        return ModelPayload(
            tracks=[
                ActorTrack(
                    samples=[
                        SpatialDetection(
                            frame_id=f.frame_id,
                            frame_index=f.frame_index,
                            timestamp_sec=f.timestamp_sec,
                            bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.4),
                        )
                    ]
                )
            ]
        )

    app = create_app(settings)
    app.state.repo.add_video(asset)
    client = TestClient(app)
    submitted = client.post(
        "/track", json={"video_id": asset.id, "prompt": "walking person", "interval": [1.0, 2.0]}
    )
    assert submitted.status_code == 202, submitted.text
    await Worker(settings, model_factory=lambda _: MockModel(responses=[response])).once()
    result = client.get("/results/" + submitted.json()["job_id"])
    assert result.status_code == 200, result.text
    assert seen == ["spatial"]
    assert result.json()["predictions"][0]["start_sec"] == 1.0
    assert result.json()["tracks"]
