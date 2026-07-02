from dspy.predict.error_localization import ErrorLocalization, localize_error


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
