from types import SimpleNamespace

from eye_processor import EyeProcessor


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
