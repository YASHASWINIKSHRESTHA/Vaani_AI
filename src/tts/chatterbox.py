"""Wrapper around Resemble AI's Chatterbox TTS (English primary model).

Runs inside its own venv (envs/chatterbox) rather than being imported
alongside the other TTS backends -- chatterbox-tts, Coqui TTS, and
transformers-based Parler-TTS commonly pin incompatible torch/transformers
versions (GAPS.md section 1b, item 6). src/router.py invokes this file's
CLI entrypoint via subprocess using that venv's interpreter.
"""

_model = None

DEFAULT_PARAMS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}
DEFAULT_SAMPLE_RATE = 24000


def _load_model():
    global _model
    if _model is None:
        from chatterbox.tts import ChatterboxTTS

        _model = ChatterboxTTS.from_pretrained(device="cuda")
    return _model


def synthesize(text: str, reference_wav: str, params: dict | None = None) -> tuple[bytes, int, dict]:
    """Zero-shot clone `reference_wav`'s voice to speak `text`.

    Returns (wav, sample_rate, params_used) -- sample_rate is the model's
    actual native output rate (read from the loaded model, not assumed),
    and params_used is logged verbatim by eval/run_eval.py so generation
    settings are reproducible, not just model identity (SPEC.md section 5).
    """
    model = _load_model()
    params_used = {**DEFAULT_PARAMS, **(params or {})}
    wav = model.generate(text, audio_prompt_path=reference_wav, **params_used)
    sample_rate = getattr(model, "sr", DEFAULT_SAMPLE_RATE)
    return wav, sample_rate, params_used


def _cli():
    import argparse
    import json

    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--params-json", default=None, help="JSON object overriding DEFAULT_PARAMS")
    args = parser.parse_args()

    params = json.loads(args.params_json) if args.params_json else None
    wav, sample_rate, params_used = synthesize(args.text, args.reference_wav, params=params)

    sf.write(args.out_wav, wav, samplerate=sample_rate)
    with open(args.out_wav + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"params": params_used, "sample_rate": sample_rate}, f)


if __name__ == "__main__":
    _cli()
