import pytest

from madewithml.serve import PredictionRequest, apply_confidence_threshold


def test_prediction_request_rejects_empty_title():
    with pytest.raises(ValueError):
        PredictionRequest(title="", description="valid description")


def test_low_confidence_prediction_routes_to_other():
    results = [{"prediction": "nlp", "probabilities": {"nlp": 0.6, "cv": 0.4}}]
    assert apply_confidence_threshold(results, threshold=0.9)[0]["prediction"] == "other"


def test_high_confidence_prediction_is_preserved():
    results = [{"prediction": "nlp", "probabilities": {"nlp": 0.95, "cv": 0.05}}]
    assert apply_confidence_threshold(results, threshold=0.9)[0]["prediction"] == "nlp"
