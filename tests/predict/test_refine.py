import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import OfferFeedback, Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        # The answer should always be one word.
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)

    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")


def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_offer_feedback_exposes_error_location_field():
    # The correction signature must now accept an externally-derived error location.
    assert "error_location" in OfferFeedback.input_fields
    assert "discussion" in OfferFeedback.output_fields


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
