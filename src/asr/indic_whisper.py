"""Round-trip ASR via IndicWhisper (Hindi) -- meaningfully lower WER on Hindi
than base Whisper large-v3, so used instead of it for the Hindi gate."""

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = "vasista22/whisper-hindi-large-v2"
        _model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
        _processor = WhisperProcessor.from_pretrained(model_name)
    return _model, _processor


def transcribe(wav_path: str) -> str:
    import soundfile as sf
    import torch

    model, processor = _load_model()
    audio, sr = sf.read(wav_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
    predicted_ids = model.generate(inputs)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
