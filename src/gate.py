"""Decide layer: deterministic pass/fail checks on a generated clip.

Runs round-trip ASR, speaker-embedding cosine similarity, and audio sanity
checks against config/thresholds.yaml. A clip only counts as a "pass" if it
clears every check here -- failures are recorded, not retried into a
good-looking number.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.similarity import speaker_cosine_similarity

THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GateResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    reason: str = ""


def audio_sanity_check(wav_path: str) -> tuple[bool, str]:
    """Checks for silence, clipping, and duration sanity. Returns (ok, reason)."""
    import soundfile as sf

    data, samplerate = sf.read(wav_path)
    if len(data) == 0:
        return False, "empty audio"
    duration_s = len(data) / samplerate
    if duration_s < 0.2:
        return False, f"duration too short ({duration_s:.2f}s)"
    peak = max(abs(data.min()), abs(data.max()))
    if peak >= 0.999:
        return False, f"clipping detected (peak={peak:.4f})"
    if peak < 0.01:
        return False, f"near-silence (peak={peak:.4f})"
    return True, ""


def gate(
    generated_wav: str,
    reference_wav: str,
    reference_text: str,
    transcribed_text: str,
    word_error_rate_pct: float,
) -> GateResult:
    """Runs every check for one generated clip and returns a single pass/fail verdict."""
    thresholds = load_thresholds()
    checks: dict = {}

    sanity_ok, sanity_reason = audio_sanity_check(generated_wav)
    checks["audio_sanity"] = {"passed": sanity_ok, "reason": sanity_reason}

    cosine = speaker_cosine_similarity(generated_wav, reference_wav)
    checks["speaker_cosine"] = {
        "value": cosine,
        "passed": cosine >= thresholds["speaker_cosine_min"],
    }

    checks["round_trip_wer"] = {
        "value": word_error_rate_pct,
        "passed": word_error_rate_pct <= thresholds["round_trip_wer_max_pct"],
    }

    all_passed = sanity_ok and checks["speaker_cosine"]["passed"] and checks["round_trip_wer"]["passed"]
    reason = "" if all_passed else "; ".join(
        name for name, c in checks.items() if not c.get("passed", True)
    )
    return GateResult(passed=all_passed, checks=checks, reason=reason)
