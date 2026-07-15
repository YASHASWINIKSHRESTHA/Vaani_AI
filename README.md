# VAANI — Multilingual Voice Pipeline

> This submission implements a per-language router ("model proposes, rules
> decide"): Chatterbox for English, XTTS-v2 for Arabic (MSA), and AI4Bharat
> Indic Parler-TTS for Hindi, each gated by a deterministic pass/fail layer
> (round-trip ASR via Whisper large-v3 / IndicWhisper, ECAPA-TDNN
> speaker-cosine similarity, and sanity checks) rather than a single
> multilingual model, because no current open-source TTS model covers all
> three languages at production quality. **[Results table and headline
> finding go here once `eval/run_eval.py` has actually been run --
> see SPEC.md section 9. Do not fill this in before real numbers exist.]**

Full design rationale, model selection, thresholds, and anti-hallucination
guardrails are in [SPEC.md](SPEC.md). Documented per-language failures are in
[FAILURES.md](FAILURES.md). Structural gaps, licensing caveats, and what a
real deployment would still need are in [LIMITATIONS.md](LIMITATIONS.md).
Pre-submission gap analysis and the run/deploy runbook are in
[GAPS.md](GAPS.md).

**Disclosure:** built with AI coding assistance for boilerplate (config
plumbing, CSV/JSON writers, wrapper scaffolding); model selection,
thresholds, eval design, and all reported numbers are mine.

**Licensing note:** Chatterbox (MIT) and AI4Bharat Indic Parler-TTS
(Apache 2.0) are commercially usable. XTTS-v2 (Coqui Public Model
License) is non-commercial only -- used here because it's the strongest
open zero-shot Arabic cloner available, not because it's deployable as-is.
See [LIMITATIONS.md](LIMITATIONS.md) section 1 for the full license table
and why open, commercially-licensed Arabic TTS is a real ecosystem gap.

## Setup

Try one shared environment first:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

If `pip install` throws a resolver conflict, that's expected -- Coqui TTS
(XTTS-v2), Chatterbox, and transformers-based Parler-TTS commonly pin
incompatible `torch`/`transformers` versions against each other. Switch to
one venv per model family instead:

```bash
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate
pip install -r requirements/base.txt                   # router, gate, similarity, ASR, eval/

python -m venv envs/chatterbox && envs/chatterbox/Scripts/pip install -r requirements/chatterbox.txt
python -m venv envs/xtts        && envs/xtts/Scripts/pip install -r requirements/xtts.txt
python -m venv envs/indic       && envs/indic/Scripts/pip install -r requirements/indic.txt
```

(`Scripts/` on Windows, `bin/` on macOS/Linux.) `src/router.py` shells out
to the right `envs/<name>` interpreter per language via `subprocess` rather
than importing all three TTS libraries into one process -- see
[GAPS.md](GAPS.md) section 1b, item 6.

## Running the eval

```bash
python -m eval.run_eval --reference-wav samples/reference/<your_voice>.wav
python -m eval.build_report
```

`run_eval.py` synthesizes every sentence in `eval/test_sentences/{en,ar,hi}.txt`
through the configured per-language model (`config/language_models.yaml`),
gates each clip against `config/thresholds.yaml`, and writes:

- a raw JSON log to `logs/` (timestamp, git commit, GPU, per-sample scores,
  and the exact generation `params` used for every clip)
- a per-sample CSV to `eval/`
- every generated `.wav` to `samples/generated/`

`build_report.py` reads the latest CSV and prints the results table --
nothing in the results section above is hand-typed.

## Collecting MOS (real listeners, not automated)

MOS requires actual humans listening -- it's the one metric nothing above
can produce. After a real eval run:

```bash
python -m eval.generate_mos_survey        # samples clips into eval/mos_survey_template.csv
# each listener duplicates the file as eval/mos_survey_<listener_id>.csv,
# fills in listener_id + score_1_to_5 (1-5) per row
python -m eval.aggregate_mos               # folds filled surveys into eval/mos_summary.csv
python -m eval.build_report                # now includes the MOS column
```

## Repo layout

See SPEC.md section 6 for the full structure and rationale.
