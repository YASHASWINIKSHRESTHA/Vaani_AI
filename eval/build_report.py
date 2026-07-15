"""Regenerates the README results table from the latest per-sample CSV.
Never hand-edit the results table -- run this script and paste its output.
"""
import argparse
import csv
import glob
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.yaml"
MOS_SUMMARY_PATH = REPO_ROOT / "eval" / "mos_summary.csv"


def latest_csv() -> Path:
    candidates = sorted(glob.glob(str(REPO_ROOT / "eval" / "results_*.csv")))
    if not candidates:
        raise FileNotFoundError("No eval/results_*.csv found -- run eval/run_eval.py first")
    return Path(candidates[-1])


def load_mos_summary() -> dict[str, dict]:
    """Reads eval/mos_summary.csv (written by eval/aggregate_mos.py) if present.
    MOS requires real human listeners -- if this file doesn't exist yet, the
    report omits the MOS column rather than inventing a number.
    """
    if not MOS_SUMMARY_PATH.exists():
        return {}
    with open(MOS_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return {row["language"]: row for row in csv.DictReader(f)}


def load_thresholds() -> dict:
    import yaml

    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_table(csv_path: Path) -> str:
    thresholds = load_thresholds()
    mos_summary = load_mos_summary()
    by_language: dict[str, list[dict]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_language.setdefault(row["language"], []).append(row)

    lines = [
        "| Language | Model | Mean WER % (n) | Mean Speaker Cosine (n) | Pass Rate (n) | MOS (n listeners) |",
        "|---|---|---|---|---|---|",
    ]
    for language, rows in sorted(by_language.items()):
        wers = [float(r["wer_pct"]) for r in rows]
        cosines = [float(r["speaker_cosine"]) for r in rows]
        n = len(rows)
        pass_rate = sum(r["passed"] == "True" for r in rows) / n
        mos_row = mos_summary.get(language)
        mos_cell = (
            f"{mos_row['mean_mos']} (n={mos_row['n_ratings']}, {mos_row['n_listeners']} listeners)"
            if mos_row else "not yet collected"
        )
        lines.append(
            f"| {language} | {rows[0]['model']} | {statistics.mean(wers):.1f} (n={n}) "
            f"| {statistics.mean(cosines):.3f} (n={n}) | {pass_rate:.0%} (n={n}) | {mos_cell} |"
        )

    lines.append("")
    lines.append(
        f"Thresholds: MOS >= {thresholds['mos_min']}, WER <= {thresholds['round_trip_wer_max_pct']}%, "
        f"speaker cosine >= {thresholds['speaker_cosine_min']}"
    )
    if not mos_summary:
        lines.append(
            "MOS not yet collected: run eval/generate_mos_survey.py, gather listener "
            "ratings, then eval/aggregate_mos.py."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to a results CSV (defaults to latest)")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    print(build_table(csv_path))


if __name__ == "__main__":
    main()
