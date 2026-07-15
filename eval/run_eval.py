"""Runs every test sentence through its language's pipeline and writes a raw
JSON log plus a per-sample CSV. This is the only thing that produces numbers
that end up in the README -- nothing in this repo is hand-typed. See
SPEC.md section 5 (anti-hallucination guardrails).
"""
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jiwer

from src.router import load_language_map, synthesize
from src.gate import gate
from src.asr import whisper as whisper_asr
from src.asr import indic_whisper
from src.wer_normalize import normalize_for_wer

REPO_ROOT = Path(__file__).resolve().parent.parent
SENTENCES_DIR = REPO_ROOT / "eval" / "test_sentences"
GENERATED_DIR = REPO_ROOT / "samples" / "generated"
REFERENCE_DIR = REPO_ROOT / "samples" / "reference"
LOGS_DIR = REPO_ROOT / "logs"

ASR_BACKENDS = {
    "whisper_large_v3": lambda wav, lang: whisper_asr.transcribe(wav, language=lang),
    "indic_whisper": lambda wav, lang: indic_whisper.transcribe(wav),
}


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def gpu_name() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "unknown (no nvidia-smi)"


def load_sentences(language: str) -> list[str]:
    path = SENTENCES_DIR / f"{language}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_language(language: str, reference_wav: str) -> list[dict]:
    language_map = load_language_map()
    asr_key = language_map[language]["asr"]
    asr_transcribe = ASR_BACKENDS[asr_key]

    sentences = load_sentences(language)
    results = []

    for i, text in enumerate(sentences):
        sample_id = f"{i:03d}"
        model_key = language_map[language]["primary"]
        out_wav = GENERATED_DIR / f"{language}_{model_key}_{sample_id}.wav"

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        meta = synthesize(
            text=text, language=language, reference_wav=reference_wav, out_wav=str(out_wav)
        )
        params_used = meta["params"]

        transcribed = asr_transcribe(str(out_wav), language)
        # Normalize before diffing -- raw comparison would count punctuation,
        # casing, and (Arabic) diacritics as pronunciation errors, inflating
        # WER on Arabic/Hindi specifically. See src/wer_normalize.py.
        wer_pct = jiwer.wer(
            normalize_for_wer(text, language), normalize_for_wer(transcribed, language)
        ) * 100

        result = gate(
            generated_wav=str(out_wav),
            reference_wav=reference_wav,
            reference_text=text,
            transcribed_text=transcribed,
            word_error_rate_pct=wer_pct,
        )

        results.append(
            {
                "sample_id": sample_id,
                "language": language,
                "model": model_key,
                "text": text,
                "transcribed": transcribed,
                "wer_pct": wer_pct,
                "speaker_cosine": result.checks["speaker_cosine"]["value"],
                "passed": result.passed,
                "fail_reason": result.reason,
                "wav_path": str(out_wav.relative_to(REPO_ROOT)),
                "params": params_used,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["en", "ar", "hi"])
    parser.add_argument("--reference-wav", required=True, help="Path to the cloned voice's reference clip")
    args = parser.parse_args()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    all_results = []
    for language in args.languages:
        all_results.extend(run_language(language, args.reference_wav))

    log = {
        "timestamp": run_timestamp,
        "git_commit": git_commit_hash(),
        "gpu": gpu_name(),
        "reference_wav": args.reference_wav,
        "results": all_results,
    }

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"eval_{run_timestamp.replace(':', '-')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # CSV keeps params as a JSON string (a raw dict isn't valid CSV cell
    # content) -- the JSON log above is the canonical, fully-structured
    # reproducibility source per SPEC.md section 5.
    csv_rows = [{**row, "params": json.dumps(row["params"], ensure_ascii=False)} for row in all_results]
    csv_path = REPO_ROOT / "eval" / f"results_{run_timestamp.replace(':', '-')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {len(all_results)} samples to {log_path} and {csv_path}")


if __name__ == "__main__":
    main()
