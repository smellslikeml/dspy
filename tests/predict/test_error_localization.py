import dspy
from dspy.predict.error_localization import ErrorLocalization, localize_error

# Importing from the non-new call-site module proves the wiring, not just the new file.
from dspy.predict.predict import Predict
from dspy.predict.refine import OfferFeedback, Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


def test_offer_feedback_exposes_error_location_field():
    # The correction signature must now accept an externally-derived error location.
    assert "error_location" in OfferFeedback.input_fields
    assert "discussion" in OfferFeedback.output_fields


def test_localize_error_flags_empty_output():
    trajectory = [
        {"module_name": "retrieve", "inputs": {}, "outputs": {"passages": ["a", "b"]}},
        {"module_name": "answer", "inputs": {}, "outputs": {"answer": ""}},
    ]
    loc = localize_error(trajectory, module_names=["retrieve", "answer"], reward_value=0.0, threshold=1.0)
    assert isinstance(loc, ErrorLocalization)
    assert loc.primary.module_name == "answer"
    assert any("empty" in reason for reason in loc.primary.reasons)


def test_localize_error_terminal_proximity_when_no_anomaly():
    trajectory = [
        {"module_name": "first", "inputs": {}, "outputs": {"x": "ok"}},
        {"module_name": "second", "inputs": {}, "outputs": {"y": "also ok"}},
    ]
    loc = localize_error(trajectory, reward_value=0.2, threshold=0.5)
    # With no objective anomaly, the terminal module (closest to the failing output) is surfaced.
    assert loc.primary.module_name == "second"


def test_localize_error_format_reports_reward_gap():
    trajectory = [{"module_name": "answer", "inputs": {}, "outputs": {"answer": "N/A"}}]
    text = localize_error(trajectory, reward_value=0.0, threshold=1.0).format()
    assert "fell short" in text
    assert "answer" in text


def test_localize_error_empty_trajectory_is_safe():
    loc = localize_error([], module_names=None, reward_value=0.0, threshold=1.0)
    assert loc.primary is None
    assert "No execution trace" in loc.format()


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


def test_refine_feeds_error_location_into_feedback_prompt():
    # Attempt 1 fails the threshold -> OfferFeedback is invoked -> attempt 2 runs.
    lm = DummyLM(
        [
            {"answer": "a rather long answer"},
            {"discussion": "The answer module was too verbose.", "advice": {"predictor": "Answer in one word."}},
            {"answer": "Brussels"},
        ]
    )
    dspy.configure(lm=lm)

    def reward_fn(kwargs, pred: Prediction) -> float:
        return 1.0 if len(pred.answer.split()) == 1 else 0.0

    predict = DummyModule("question -> answer", lambda self, **kw: self.predictor(**kw))
    refine = Refine(module=predict, N=2, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result is not None
    # The OfferFeedback prompt must carry the trace-derived error location, proving the integration fired.
    feedback_prompts = [str(entry["messages"]) for entry in lm.history if "error_location" in str(entry["messages"])]
    assert feedback_prompts, "OfferFeedback prompt should include the error_location input field"
    assert any("Objective trace analysis" in p for p in feedback_prompts)
