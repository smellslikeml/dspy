"""Self-summarization compaction for RLM REPL trajectories.

``RLM.forward`` re-feeds the full :class:`~dspy.primitives.repl_types.REPLHistory`
to the model on every iteration. On long recursive runs this grows without bound
and eventually exhausts the context budget. :class:`REPLHistoryCompactor` lets the
model compress its own trajectory: once the rendered history exceeds a size
threshold, the oldest steps are replaced by a compact, LM-written summary while
the most recent steps are kept verbatim.

This implements the *self-summarization* contract from ReSum (Wu et al., 2025,
arxiv.org/abs/2606.13316): grow trajectory in -> compact summary -> continue. We
deliver the inference-time memory-management result, not the paper's RL training
loop (the summarization-aware adaptive rollout and advantage), which is out of
scope for an inference framework.
"""

from __future__ import annotations

from typing import Callable

from dspy.predict.predict import Predict
from dspy.primitives.repl_types import REPLHistory
from dspy.signatures import InputField, OutputField, Signature

__all__ = ["TrajectorySummary", "REPLHistoryCompactor"]


class TrajectorySummary(Signature):
    """Summarize a REPL reasoning trajectory so the agent can keep working without re-reading it.

    Preserve everything needed to continue: concrete findings, computed values
    (IDs, numbers, intermediate results), what has already been tried, and what
    remains to be done. Be faithful and concise; do not invent results."""

    prior_summary: str = InputField(desc="Summary of even earlier steps, if any (may be empty).")
    trajectory: str = InputField(desc="The earlier REPL steps (reasoning, code, output) being compacted away.")
    summary: str = OutputField(desc="A compact recap that supersedes both prior_summary and trajectory.")


class REPLHistoryCompactor:
    """Compact a growing :class:`REPLHistory` via model self-summarization.

    Args:
        max_history_chars: Compact once the rendered history exceeds this many
            characters. Sized below the model's context budget.
        keep_recent: Number of most-recent entries to keep verbatim; older
            entries are folded into the summary.
        summarize_fn: Optional ``(prior_summary, trajectory) -> str`` callable.
            Defaults to a :class:`dspy.Predict` over :class:`TrajectorySummary`,
            which uses the configured ``dspy.settings.lm``.

    Example:
        ```python
        compactor = dspy.REPLHistoryCompactor(max_history_chars=40_000, keep_recent=3)
        rlm = dspy.RLM("context, query -> answer", max_iterations=40, compactor=compactor)
        ```
    """

    def __init__(
        self,
        max_history_chars: int = 40_000,
        keep_recent: int = 3,
        summarize_fn: Callable[[str, str], str] | None = None,
    ):
        if max_history_chars <= 0:
            raise ValueError("max_history_chars must be positive")
        if keep_recent < 0:
            raise ValueError("keep_recent must be non-negative")
        self.max_history_chars = max_history_chars
        self.keep_recent = keep_recent
        self._summarize_fn = summarize_fn
        self._summarizer: Predict | None = None

    def estimate_size(self, history: REPLHistory) -> int:
        """Estimate the rendered size of the history in characters."""
        return len(history.format())

    def should_compact(self, history: REPLHistory) -> bool:
        """Whether the history is large enough to warrant compaction."""
        return len(history.entries) > self.keep_recent and self.estimate_size(history) > self.max_history_chars

    def _summarize(self, prior_summary: str, trajectory: str) -> str:
        if self._summarize_fn is not None:
            return self._summarize_fn(prior_summary, trajectory)
        if self._summarizer is None:
            self._summarizer = Predict(TrajectorySummary)
        return self._summarizer(prior_summary=prior_summary, trajectory=trajectory).summary

    def compact(self, history: REPLHistory) -> REPLHistory:
        """Return a new history with the oldest steps folded into a summary.

        Steps beyond ``keep_recent`` (and any existing summary) are summarized
        into a single recap. If there is nothing to fold, the history is
        returned unchanged.
        """
        if len(history.entries) <= self.keep_recent:
            return history

        split = len(history.entries) - self.keep_recent
        older = history.entries[:split]
        recent = history.entries[split:]

        trajectory = "\n".join(
            entry.format(index=i, max_output_chars=history.max_output_chars) for i, entry in enumerate(older)
        )
        summary = self._summarize(history.summary, trajectory)

        return REPLHistory(entries=recent, max_output_chars=history.max_output_chars, summary=summary)
