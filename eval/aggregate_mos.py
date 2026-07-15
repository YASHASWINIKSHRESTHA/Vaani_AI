"""Aggregates filled-in MOS survey CSVs (eval/mos_survey_<listener_id>.csv)
into a per-language mean and writes eval/mos_summary.csv, which
build_report.py reads to fold MOS into the results table.

Never hand-type a MOS number into README -- this script and real listener
CSVs are the only source.
"""
import argparse
import csv
import glob
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
OUT_PATH = EVAL_DIR / "mos_summary.csv"


def load_filled_surveys() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(EVAL_DIR / "mos_survey_*.csv"))):
        if Path(path).name == "mos_survey_template.csv":
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("score_1_to_5", "").strip() and row.get("listener_id", "").strip():
                    rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    by_language: dict[str, list[dict]] = {}
    for row in rows:
        by_language.setdefault(row["language"], []).append(row)

    summary = []
    for language, lang_rows in sorted(by_language.items()):
        scores = [float(r["score_1_to_5"]) for r in lang_rows]
        listeners = {r["listener_id"] for r in lang_rows}
        summary.append({
            "language": language,
            "mean_mos": round(statistics.mean(scores), 2),
            "n_ratings": len(scores),
            "n_listeners": len(listeners),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    rows = load_filled_surveys()
    if not rows:
        raise FileNotFoundError(
            f"No filled-in eval/mos_survey_<listener_id>.csv files with scores found in {EVAL_DIR} -- "
            "run eval/generate_mos_survey.py, have listeners fill in copies, then re-run this."
        )

    summary = build_summary(rows)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "mean_mos", "n_ratings", "n_listeners"])
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        print(
            f"{row['language']}: MOS {row['mean_mos']} "
            f"(n={row['n_ratings']} ratings, {row['n_listeners']} listeners)"
        )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
