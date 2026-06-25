"""Objective error localization from a program's execution trajectory.

Motivated by Tyen et al., "LLMs cannot find reasoning errors, but can correct
them given the error location" (https://arxiv.org/abs/2311.08516). The paper
shows that LLM self-correction degrades mainly because models are unreliable at
*finding* where a mistake occurred, not at *fixing* a mistake once its location
is known.

This module derives a candidate error location from objective signals already
present in a DSPy execution trace (empty/missing outputs, error-like content,
and proximity to the failing program output) rather than asking the LM to
self-diagnose blame from scratch. The resulting localization is fed to the
correction step so the model can spend its effort on fixing rather than on the
finding task it does poorly.
"""

from dataclasses import dataclass, field

# Substrings that, when they dominate an output value, suggest the module
# emitted an error/non-answer rather than a real result.
_ERROR_MARKERS = (
    "traceback",
    "exception",
    "error:",
    "n/a",
    "i cannot",
    "i can't",
    "unable to",
    "i'm sorry",
)


@dataclass
class ModuleSuspicion:
    """A single module's localization score and the reasons behind it."""

    module_name: str
    step_index: int
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class ErrorLocalization:
    """Ranked candidate error locations derived from an execution trajectory."""

    suspects: list[ModuleSuspicion]
    reward_value: float | None
    threshold: float | None

    @property
    def primary(self) -> ModuleSuspicion | None:
        """The most suspect module, or None if the trajectory was empty."""
        return self.suspects[0] if self.suspects else None

    def format(self) -> str:
        """Render a concise localization hint for the correction step."""
        if not self.suspects:
            return "No execution trace was available to localize the error."

        gap = None
        if self.reward_value is not None and self.threshold is not None:
            gap = self.threshold - self.reward_value

        lines = []
        if gap is not None:
            lines.append(
                f"The reward ({self.reward_value:g}) fell short of the threshold ({self.threshold:g}) by {gap:g}."
            )
        lines.append(
            "Objective trace analysis flags the following module(s) as the most "
            "likely error location. Focus your correction there first; treat "
            "other modules as correct unless the trace contradicts that."
        )
        for rank, suspect in enumerate(self.suspects, start=1):
            reasons = "; ".join(suspect.reasons) if suspect.reasons else "closest to the failing program output"
            lines.append(f"  {rank}. {suspect.module_name}: {reasons}")
        return "\n".join(lines)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _looks_like_error(value) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().lower()
    if not stripped:
        return False
    return any(marker in stripped for marker in _ERROR_MARKERS)


def localize_error(
    trajectory: list[dict],
    module_names: list[str] | None = None,
    reward_value: float | None = None,
    threshold: float | None = None,
    top_k: int = 2,
) -> ErrorLocalization:
    """Rank trajectory steps by how likely they are to be the error location.

    Args:
        trajectory: Per-module execution records, each a dict with at least
            ``module_name`` and ``outputs`` (as produced by ``Refine.forward``).
        module_names: Optional list of all module names; ensures modules that
            never produced a usable output are still considered.
        reward_value: The scalar reward assigned to the program's outputs.
        threshold: The target reward threshold the program failed to reach.
        top_k: Maximum number of suspect modules to surface.

    Returns:
        An ``ErrorLocalization`` with suspects ranked by descending score. Ties
        are broken toward earlier steps, since errors propagate forward and the
        earliest anomalous module is the root cause to correct.
    """
    suspicions: list[ModuleSuspicion] = []
    n_steps = len(trajectory)

    for step_index, record in enumerate(trajectory):
        name = record.get("module_name", f"step_{step_index}")
        outputs = record.get("outputs") or {}
        suspicion = ModuleSuspicion(module_name=name, step_index=step_index)

        if isinstance(outputs, dict):
            empty_fields = [k for k, v in outputs.items() if _is_empty(v)]
            error_fields = [k for k, v in outputs.items() if _looks_like_error(v)]
            if not outputs:
                suspicion.score += 2.0
                suspicion.reasons.append("produced no output")
            if empty_fields:
                suspicion.score += 2.0
                suspicion.reasons.append(f"left output field(s) empty: {', '.join(empty_fields)}")
            if error_fields:
                suspicion.score += 1.5
                suspicion.reasons.append(f"emitted error-like content in: {', '.join(error_fields)}")
        elif _is_empty(outputs):
            suspicion.score += 2.0
            suspicion.reasons.append("produced no output")

        # The terminal module produces the program output that the reward judged,
        # so it sits closest to the observed failure.
        if step_index == n_steps - 1:
            suspicion.score += 1.0

        suspicions.append(suspicion)

    # Modules that never appear in the trajectory could not have run to completion.
    if module_names:
        seen = {s.module_name for s in suspicions}
        for name in module_names:
            if name not in seen:
                suspicions.append(
                    ModuleSuspicion(
                        module_name=name,
                        step_index=n_steps,
                        score=1.0,
                        reasons=["did not appear in the execution trace"],
                    )
                )

    # Highest score first; earlier steps win ties (propagation root cause).
    suspicions.sort(key=lambda s: (-s.score, s.step_index))
    suspects = [s for s in suspicions if s.score > 0][:top_k]

    # Fall back to the top-ranked step if nothing crossed the score floor.
    if not suspects and suspicions:
        suspects = suspicions[:1]

    return ErrorLocalization(suspects=suspects, reward_value=reward_value, threshold=threshold)
