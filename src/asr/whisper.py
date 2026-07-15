"""Round-trip ASR via Whisper large-v3 (English, Arabic)."""

_model = None


def _load_model():
    global _model
    if _model is None:
        import whisper

        _model = whisper.load_model("large-v3")
    return _model


def transcribe(wav_path: str, language: str) -> str:
    model = _load_model()
    result = model.transcribe(wav_path, language=language)
    return result["text"].strip()
