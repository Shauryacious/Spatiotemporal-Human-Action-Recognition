import pytest
from conftest import frame

from shar.errors import SharError
from shar.inference.multimodal.base import ModelRequest, ModelResponse
from shar.inference.validation import validate_response


@pytest.mark.parametrize(
    "payload",
    [
        "bad json",
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
