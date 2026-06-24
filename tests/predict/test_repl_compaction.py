"""Tests for REPL trajectory self-summarization (ReSum-style compaction).

Covers the compactor itself plus its wiring into the existing ``RLM.forward``
loop, so these tests import from the non-new ``dspy.predict.rlm`` and
``dspy.primitives.repl_types`` modules as well as the new compaction module.
"""

from dspy.predict.repl_compaction import REPLHistoryCompactor, TrajectorySummary
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.repl_types import REPLEntry, REPLHistory
from tests.mock_interpreter import MockInterpreter


def make_mock_predictor(responses):
    """Minimal scripted predictor (mirrors the helper in test_rlm.py)."""

    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**result)

    return MockPredictor()


def _history(n):
    history = REPLHistory()
    for i in range(n):
        history = history.append(reasoning=f"step {i}", code=f"print({i})", output=f"out {i}")
    return history


# ---------------------------------------------------------------------------
# REPLHistory summary rendering (existing repl_types module)
# ---------------------------------------------------------------------------


def test_summary_renders_ahead_of_entries():
    history = REPLHistory(
        entries=[REPLEntry(reasoning="latest", code="print('x')", output="x")],
        summary="Earlier I found the magic number is 42.",
    )
    rendered = history.format()
    assert "=== Summary of earlier steps ===" in rendered
    assert "magic number is 42" in rendered
    # Summary appears before the surviving step.
    assert rendered.index("Summary of earlier steps") < rendered.index("Step 1")


def test_append_preserves_summary():
    history = REPLHistory(summary="prior recap").append(code="print(1)", output="1")
    assert history.summary == "prior recap"
    assert len(history.entries) == 1


def test_empty_history_message_unchanged():
    assert "have not interacted" in REPLHistory().format()


# ---------------------------------------------------------------------------
# REPLHistoryCompactor unit behavior
# ---------------------------------------------------------------------------


def test_should_compact_thresholds():
    history = _history(5)
    big = REPLHistoryCompactor(max_history_chars=1_000_000, keep_recent=1)
    assert big.should_compact(history) is False  # under size threshold

    tiny = REPLHistoryCompactor(max_history_chars=10, keep_recent=1)
    assert tiny.should_compact(history) is True

    # Not enough entries to fold beyond keep_recent.
    assert tiny.should_compact(_history(1)) is False


def test_compact_folds_old_entries_into_summary():
    calls = []

    def fake_summarize(prior, trajectory):
        calls.append((prior, trajectory))
        return "RECAP"

    compactor = REPLHistoryCompactor(max_history_chars=10, keep_recent=2, summarize_fn=fake_summarize)
    compacted = compactor.compact(_history(5))

    # Only the most recent `keep_recent` entries survive verbatim.
    assert len(compacted.entries) == 2
    assert compacted.summary == "RECAP"
    assert compacted.entries[0].output == "out 3"
    assert compacted.entries[1].output == "out 4"

    # The folded (older) steps were handed to the summarizer.
    prior, trajectory = calls[0]
    assert prior == ""
    assert "out 0" in trajectory and "out 2" in trajectory
    assert "out 4" not in trajectory


def test_compact_accumulates_prior_summary():
    seen_priors = []

    def fake_summarize(prior, trajectory):
        seen_priors.append(prior)
        return f"summary#{len(seen_priors)}"

    compactor = REPLHistoryCompactor(max_history_chars=10, keep_recent=1, summarize_fn=fake_summarize)
    once = compactor.compact(_history(3))
    assert once.summary == "summary#1"

    # Feeding it back in (with new entries) carries the previous summary forward.
    grown = once.append(code="print('new')", output="new")
    twice = compactor.compact(grown)
    assert twice.summary == "summary#2"
    assert seen_priors[1] == "summary#1"


def test_compact_noop_when_few_entries():
    compactor = REPLHistoryCompactor(keep_recent=3, summarize_fn=lambda p, t: "X")
    history = _history(2)
    assert compactor.compact(history) is history


def test_trajectory_summary_signature_fields():
    assert "prior_summary" in TrajectorySummary.input_fields
    assert "trajectory" in TrajectorySummary.input_fields
    assert "summary" in TrajectorySummary.output_fields


# ---------------------------------------------------------------------------
# Wiring: RLM.forward actually invokes the compactor on long runs
# ---------------------------------------------------------------------------


def test_rlm_forward_compacts_growing_history():
    summarize_calls = []

    def fake_summarize(prior, trajectory):
        summarize_calls.append((prior, trajectory))
        return "COMPACT SUMMARY"

    compactor = REPLHistoryCompactor(max_history_chars=10, keep_recent=1, summarize_fn=fake_summarize)

    mock = MockInterpreter(responses=["out1", "out2", "out3", FinalOutput({"answer": "done"})])
    rlm = RLM("query -> answer", max_iterations=6, interpreter=mock, compactor=compactor)
    rlm.generate_action = make_mock_predictor(
        [
            {"reasoning": "explore", "code": "print('a')"},
            {"reasoning": "explore", "code": "print('b')"},
            {"reasoning": "explore", "code": "print('c')"},
            {"reasoning": "finish", "code": 'SUBMIT("done")'},
        ]
    )

    result = rlm.forward(query="test")

    assert result.answer == "done"
    # Compaction ran at least once during the loop.
    assert len(summarize_calls) >= 1
    # Cumulative: a later compaction sees the earlier summary as prior context.
    if len(summarize_calls) >= 2:
        assert summarize_calls[1][0] == "COMPACT SUMMARY"
    # Final trajectory is compacted, not the full step-by-step history.
    assert len(result.trajectory) <= 2


def test_rlm_forward_without_compactor_keeps_full_history():
    mock = MockInterpreter(responses=["out1", "out2", FinalOutput({"answer": "done"})])
    rlm = RLM("query -> answer", max_iterations=6, interpreter=mock)
    rlm.generate_action = make_mock_predictor(
        [
            {"reasoning": "explore", "code": "print('a')"},
            {"reasoning": "explore", "code": "print('b')"},
            {"reasoning": "finish", "code": 'SUBMIT("done")'},
        ]
    )

    result = rlm.forward(query="test")
    assert result.answer == "done"
    # No compaction: every step survives in the trajectory.
    assert len(result.trajectory) == 3
