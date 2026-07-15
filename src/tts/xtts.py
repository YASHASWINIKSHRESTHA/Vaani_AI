"""Wrapper around Coqui XTTS-v2 (Arabic/MSA primary model).

Runs inside its own venv (envs/xtts) -- see src/tts/chatterbox.py's module
docstring for why. src/router.py invokes this file's CLI entrypoint via
subprocess using that venv's interpreter.
"""

_model = None

DEFAULT_PARAMS = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
}
DEFAULT_SAMPLE_RATE = 24000


def _load_model():
    global _model
    if _model is None:
        from TTS.api import TTS

        _model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    return _model


def synthesize(
    text: str, reference_wav: str, language: str = "ar", params: dict | None = None
) -> tuple[bytes, int, dict]:
    """Zero-shot clone `reference_wav`'s voice to speak `text` in `language`.

    Returns (wav, sample_rate, params_used) -- see chatterbox.py for why.
    """
    model = _load_model()
    params_used = {**DEFAULT_PARAMS, **(params or {})}
    wav = model.tts(text=text, speaker_wav=reference_wav, language=language, **params_used)
    sample_rate = getattr(
        getattr(model, "synthesizer", None), "output_sample_rate", DEFAULT_SAMPLE_RATE
    )
    return wav, sample_rate, params_used


def _cli():
    import argparse
    import json

    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--language", default="ar")
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--params-json", default=None, help="JSON object overriding DEFAULT_PARAMS")
    args = parser.parse_args()

    params = json.loads(args.params_json) if args.params_json else None
    wav, sample_rate, params_used = synthesize(
        args.text, args.reference_wav, language=args.language, params=params
    )

    sf.write(args.out_wav, wav, samplerate=sample_rate)
    with open(args.out_wav + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"params": params_used, "sample_rate": sample_rate}, f)


if __name__ == "__main__":
    _cli()
