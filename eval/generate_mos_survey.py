"""Builds a MOS survey CSV from real generated clips in samples/generated/.

The brief requires Mean Opinion Score from real human listeners -- nothing
here can be automated, so this script only prepares the listening sheet.
Send the output file to yourself + a few others; each listener fills in
their own `listener_id` and `score_1_to_5` (1-5) per row and saves it as
`eval/mos_survey_<listener_id>.csv`. Run `eval/aggregate_mos.py` once those
filled-in copies exist.
"""
import argparse
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "samples" / "generated"
OUT_PATH = REPO_ROOT / "eval" / "mos_survey_template.csv"


def clips_by_language() -> dict[str, list[Path]]:
    by_language: dict[str, list[Path]] = {}
    for wav_path in sorted(GENERATED_DIR.glob("*.wav")):
        language = wav_path.name.split("_", 1)[0]
        by_language.setdefault(language, []).append(wav_path)
    return by_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", type=int, default=8, help="Max clips to sample per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_language = clips_by_language()
    if not by_language:
        raise FileNotFoundError(
            f"No .wav files found in {GENERATED_DIR} -- run eval/run_eval.py first"
        )

    rng = random.Random(args.seed)
    rows = []
    for language, clips in sorted(by_language.items()):
        chosen = clips if len(clips) <= args.per_language else rng.sample(clips, args.per_language)
        for clip in sorted(chosen):
            rows.append({
                "clip_path": str(clip.relative_to(REPO_ROOT)),
                "language": language,
                "listener_id": "",
                "score_1_to_5": "",
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "language", "listener_id", "score_1_to_5"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows across {len(by_language)} languages to {OUT_PATH}")
    print("Duplicate this file per listener as eval/mos_survey_<listener_id>.csv, "
          "fill in listener_id + score_1_to_5, then run eval/aggregate_mos.py")


if __name__ == "__main__":
    main()
