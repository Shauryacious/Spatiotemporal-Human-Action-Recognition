import pytest
from pydantic import ValidationError

from shar.schemas import ActionPrediction, BBox, BoundaryEstimate


@pytest.mark.parametrize(
    "values",
    [
        dict(start_sec=2.0, end_sec=1.0),
        dict(start_sec=-1.0, end_sec=1.0),
        dict(start_sec=0.0, end_sec=float("nan")),
        dict(start_sec="0", end_sec=1.0),
    ],
)
def test_bad_intervals(values):
    with pytest.raises(ValidationError):
        ActionPrediction(label="walking", **values)


def test_box_and_boundary():
    with pytest.raises(ValidationError):
        BBox(x=0.9, y=0.0, width=0.2, height=0.2)
    with pytest.raises(ValidationError):
        BoundaryEstimate(estimate=1.0, lower=2.0, upper=3.0)
    with pytest.raises(ValidationError):
        ActionPrediction(label="   ", start_sec=0.0, end_sec=1.0)
