import pytest
from conftest import frame

from shar.errors import SharError
from shar.inference.multimodal.base import ModelRequest, ModelResponse
from shar.inference.validation import validate_response


@pytest.mark.parametrize(
    "payload",
    [
        "bad json",
        {"predictions": [], "unexpected": "still rejected locally"},
        {
            "predictions": [
                {"label": "walk", "start_sec": 0.0, "end_sec": 2.0, "evidence_frame_ids": ["f0"]}
            ]
        },
        {"recognition": [{"label": "walk", "evidence_frame_ids": ["foreign"]}]},
        {
            "predictions": [
                {"label": "walk", "start_sec": 0.0, "end_sec": 1.0, "evidence_frame_ids": []}
            ]
        },
    ],
)
def test_bad_response(payload):
    with pytest.raises(SharError):
        validate_response(
            ModelResponse(payload, "mock"), ModelRequest("temporal", "walk", [frame()], 0.0, 1.0)
        )


def test_refusal_not_empty():
    with pytest.raises(SharError, match="complete"):
        validate_response(
            ModelResponse({}, "mock", finish_reason="SAFETY"),
            ModelRequest("temporal", "walk", [frame()], 0.0, 1.0),
        )


def test_spatial_schema_does_not_request_temporal_boundaries():
    from shar.inference import prompts

    schema = prompts.output_schema("spatial")
    assert set(schema["properties"]) == {"tracks"}
    assert "ActionPrediction" not in schema["$defs"]
    assert "BoundaryEstimate" not in schema["$defs"]
    assert "SpatialDetection" in schema["$defs"]
    assert set(prompts.output_schema("temporal")["properties"]) >= {"predictions", "tracks"}


def test_spatial_tracks_validate_with_wider_input_boundary_uncertainty():
    # The temporal uncertainty bracket precedes this spatial sampling interval.
    # It belongs in input context, not in the spatial model's output.
    context = {
        "prediction": {
            "actor_ids": [],
            "start_boundary": {
                "estimate": 1.25,
                "lower": 1.0,
                "upper": 1.25,
            },
        }
    }
    request = ModelRequest("spatial", "wave", [frame(0, 1.25)], 1.25, 3.0, context=context)
    payload = {
        "tracks": [
            {
                "samples": [
                    {
                        "frame_id": "f0",
                        "frame_index": 0,
                        "timestamp_sec": 1.25,
                        "bbox": {"x": 0.2, "y": 0.1, "width": 0.5, "height": 0.8},
                    }
                ]
            }
        ]
    }
    assert len(validate_response(ModelResponse(payload, "mock"), request).tracks) == 1
