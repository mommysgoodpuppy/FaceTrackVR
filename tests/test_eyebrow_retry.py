from types import SimpleNamespace

from eye_processor import EyeProcessor
from eyebrow import EyeBrow


def _processor(variant="BSB"):
    return SimpleNamespace(
        settings=SimpleNamespace(gui_model_variant=variant),
        eyebrow_runner=None,
        _eyebrow_failed_variant=None,
    )


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
