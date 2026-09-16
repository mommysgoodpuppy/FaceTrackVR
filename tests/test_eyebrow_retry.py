from types import MethodType, SimpleNamespace
from unittest import mock

from eye_processor import EyeProcessor
from eyebrow import EyeBrow


def _processor(variant="BSB"):
    return SimpleNamespace(
        settings=SimpleNamespace(gui_model_variant=variant),
        eyebrow_runner=None,
        _eyebrow_failed_variant=None,
    )


def _real_processor(*, next_enabled, eyebrow_enabled, runner=None):
    processor = object.__new__(EyeProcessor)
    processor.settings = SimpleNamespace(
        gui_NEXT=next_enabled,
        gui_NEXT_BSB=False,
        gui_eyebrow=eyebrow_enabled,
    )
    processor.eyebrow_runner = runner
    processor._eyebrow_failed_variant = "BSB"
    processor._run_eyebrow = MethodType(mock.Mock(), processor)
    return processor


def test_failed_eyebrow_model_is_not_retried_every_frame(monkeypatch):
    attempts = []

    def fail(variant):
        attempts.append(variant)
        raise FileNotFoundError(variant)

    monkeypatch.setattr("eye_processor.EyeBrow", fail)
    processor = _processor()

    EyeProcessor._run_eyebrow(processor, object())
    EyeProcessor._run_eyebrow(processor, object())

    assert attempts == ["BSB"]
    assert processor._eyebrow_failed_variant == "BSB"


def test_changing_variant_allows_one_new_initialization_attempt(monkeypatch):
    attempts = []

    def fail(variant):
        attempts.append(variant)
        raise FileNotFoundError(variant)

    monkeypatch.setattr("eye_processor.EyeBrow", fail)
    processor = _processor()

    EyeProcessor._run_eyebrow(processor, object())
    processor.settings.gui_model_variant = "ETVR"
    EyeProcessor._run_eyebrow(processor, object())
    EyeProcessor._run_eyebrow(processor, object())

    assert attempts == ["BSB", "ETVR"]
    assert processor._eyebrow_failed_variant == "ETVR"


def test_missing_variant_model_falls_back_to_bundled_etvr_model(monkeypatch):
    loaded = []

    class Session:
        def __init__(self, path, *_args, **_kwargs):
            loaded.append(path)

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

    monkeypatch.setattr(
        "eyebrow.resource_path", lambda path: f"/bundle/{path}"
    )
    monkeypatch.setattr(
        "eyebrow.os.path.isfile",
        lambda path: path.endswith("Eyebrow_ETVR.onnx"),
    )
    monkeypatch.setattr("eyebrow.onnxruntime.InferenceSession", Session)

    runner = EyeBrow("BSB")
    try:
        assert loaded == ["/bundle/Models/Eyebrow_ETVR.onnx"]
        assert runner.variant == "BSB"
    finally:
        runner.stop()


def test_next_disables_and_stops_standalone_eyebrow_model():
    runner = mock.Mock()
    processor = _real_processor(
        next_enabled=True,
        eyebrow_enabled=True,
        runner=runner,
    )

    processor._update_standalone_eyebrow(object())

    runner.stop.assert_called_once_with()
    assert processor.eyebrow_runner is None
    assert processor._eyebrow_failed_variant is None
    processor._run_eyebrow.__func__.assert_not_called()


def test_non_next_tracker_can_run_standalone_eyebrow_model():
    processor = _real_processor(
        next_enabled=False,
        eyebrow_enabled=True,
    )
    frame = object()

    processor._update_standalone_eyebrow(frame)

    processor._run_eyebrow.__func__.assert_called_once_with(processor, frame)
