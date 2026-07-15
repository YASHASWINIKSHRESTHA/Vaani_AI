"""Speaker-embedding cosine similarity via SpeechBrain's ECAPA-TDNN."""

# spkrec-ecapa-voxceleb is trained on 16kHz audio. XTTS-v2, Chatterbox, and
# Indic Parler-TTS output at different native sample rates (commonly
# 24kHz/44.1kHz), and the reference clip may be recorded at yet another rate.
# Without resampling everything to this rate first, the cosine number partly
# measures a sample-rate artifact rather than voice similarity (GAPS.md
# section 1b, item 7).
TARGET_SAMPLE_RATE = 16000

_classifier = None


def _load_model():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    return _classifier


def _load_at_target_rate(wav_path: str):
    import torchaudio

    signal, sample_rate = torchaudio.load(wav_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        signal = torchaudio.functional.resample(
            signal, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
    return signal


def speaker_cosine_similarity(wav_path_a: str, wav_path_b: str) -> float:
    import torch.nn.functional as F

    classifier = _load_model()

    signal_a = _load_at_target_rate(wav_path_a)
    signal_b = _load_at_target_rate(wav_path_b)

    embedding_a = classifier.encode_batch(signal_a).squeeze()
    embedding_b = classifier.encode_batch(signal_b).squeeze()

    return F.cosine_similarity(embedding_a, embedding_b, dim=0).item()
