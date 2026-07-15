"""Wrapper around AI4Bharat Indic Parler-TTS (Hindi primary model).

Runs inside its own venv (envs/indic) -- see src/tts/chatterbox.py's module
docstring for why. src/router.py invokes this file's CLI entrypoint via
subprocess using that venv's interpreter.
"""

_model = None
_tokenizer = None
_description_tokenizer = None

DEFAULT_VOICE_DESCRIPTION = "A clear, natural Hindi voice."
DEFAULT_SAMPLE_RATE = 44100


def _load_model():
    global _model, _tokenizer, _description_tokenizer
    if _model is None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        model_name = "ai4bharat/indic-parler-tts"
        _model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to("cuda")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _description_tokenizer = AutoTokenizer.from_pretrained(_model.config.text_encoder._name_or_path)
    return _model, _tokenizer, _description_tokenizer


def synthesize(
    text: str,
    reference_wav: str,
    voice_description: str = DEFAULT_VOICE_DESCRIPTION,
    params: dict | None = None,
) -> tuple[bytes, int, dict]:
    """Generate Hindi speech. Indic Parler-TTS is description-conditioned, not
    zero-shot-cloned from `reference_wav` directly -- reference_wav is kept in
    the signature for interface parity with the other backends and is used
    only for the downstream speaker-similarity gate, not as a cloning input.

    Returns (wav_array, sample_rate, params_used) -- see chatterbox.py for why.
    """
    model, tokenizer, description_tokenizer = _load_model()
    generation_kwargs = params or {}
    params_used = {"voice_description": voice_description, **generation_kwargs}

    input_ids = description_tokenizer(voice_description, return_tensors="pt").input_ids.to("cuda")
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, **generation_kwargs)
    sample_rate = getattr(model.config, "sampling_rate", DEFAULT_SAMPLE_RATE)
    return generation.cpu().numpy().squeeze(), sample_rate, params_used


def _cli():
    import argparse
    import json

    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference-wav", required=True)
    parser.add_argument("--out-wav", required=True)
    parser.add_argument("--voice-description", default=DEFAULT_VOICE_DESCRIPTION)
    parser.add_argument("--params-json", default=None, help="JSON object of extra model.generate() kwargs")
    args = parser.parse_args()

    params = json.loads(args.params_json) if args.params_json else None
    wav, sample_rate, params_used = synthesize(
        args.text, args.reference_wav, voice_description=args.voice_description, params=params
    )

    sf.write(args.out_wav, wav, samplerate=sample_rate)
    with open(args.out_wav + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"params": params_used, "sample_rate": sample_rate}, f)


if __name__ == "__main__":
    _cli()
