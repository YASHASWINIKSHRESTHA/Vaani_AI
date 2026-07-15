# VAANI — Pre-Submission Gap Analysis, Hallucination Audit, and Runbook

Read alongside SPEC.md. This file exists to be run *against*, not just read once.

---

## 1. Gaps vs. the brief — ranked by how much they'd cost you if missed

### 1a. Blocking (submission is incomplete without these)

1. **No MOS collection mechanism exists yet.** The brief requires "Mean Opinion Score... from real listeners (you plus a few others)" — this is not something `run_eval.py` can produce on its own; it needs actual humans listening. Nothing in the current scaffold does this. Build a `eval/mos_survey_template.csv` (columns: `clip_path, language, listener_id, score_1_to_5`) and a `eval/aggregate_mos.py` that reads filled-in CSVs and folds MOS into the same report `build_report.py` produces. Without this, metric #1 in Section 3 is simply absent from your submission — an automatic gap against their own rubric.

2. **No "what's still missing / how you'd improve it" section.** This is an explicit, separate bullet in Section 5's deliverable list, distinct from failure modes. Add a `LIMITATIONS.md` (or a section in README) that names concrete open-source gaps — see 1c below for a genuinely strong one you can use.

3. **License check not done.** Ground rules require open-source models for core generation. Verify licenses before you submit, not after:
   - Chatterbox (Resemble AI): MIT — clean.
   - **XTTS-v2 (Coqui): Coqui Public Model License (CPML) — non-commercial only.** This is real and easy to miss. It doesn't block using it for a take-home evaluation, but it means XTTS-v2 is **not** a deployable choice for an actual product without a commercial license. State this explicitly — it's exactly the kind of "what's missing in open source" finding the brief rewards, not a liability to hide.
   - AI4Bharat Indic Parler-TTS: Apache 2.0 — clean.
   - Meta MMS-TTS (your Arabic fallback-of-a-fallback in SPEC section 4): CC-BY-NC — also non-commercial. Worth noting: **commercially-licensed open Arabic TTS is genuinely thin right now** — that's a real, defensible "what's missing" finding, not a filler sentence.

4. **Generation parameters aren't logged, only model identity.** "Reproducibility: name your model versions, hardware, and key parameters" (Section 6) means temperature/speed/sampling settings passed to each TTS call need to land in the JSON log too, not just the model's HF revision hash. Add a `params: {...}` field to whatever dict `run_eval.py` writes per sample.

5. **No disclosure line for AI-assisted development.** Ground rules explicitly permit it ("Using AI coding assistants is fine and expected") — but stating it plainly in the README ("built with AI coding assistance for boilerplate; model selection, thresholds, and eval design are mine") costs one sentence and preempts the exact suspicion the brief is designed to detect.

### 1b. Real risk, not yet hit because nothing's been run

6. **Dependency conflicts across the three TTS libraries are likely, not hypothetical.** Coqui TTS (XTTS-v2), Chatterbox, and AI4Bharat's transformers-based Indic Parler-TTS commonly pin incompatible `torch`/`transformers` versions against each other. A single shared `requirements.txt` for all three is the most probable first thing to break. See Runbook section 2 for the fix — don't discover this by trial and error against the clock.

7. **Sample-rate mismatches will silently corrupt your cosine-similarity numbers.** XTTS-v2, Chatterbox, and Indic Parler-TTS don't necessarily output at the same sample rate, and ECAPA-TDNN expects 16kHz input. If `similarity.py` doesn't resample everything to a consistent rate before embedding, you'll get a cosine number that's measuring a sample-rate artifact, not voice similarity — and it'll look like a model failure when it's actually a bug. Verify resampling happens before you trust any similarity number.

8. **Naive WER will overstate failure on Arabic and Hindi specifically.** Raw ASR transcript vs. input text, compared without normalization, will get dinged for punctuation, diacritics (Arabic), and transliteration variance (Hindi) that have nothing to do with the TTS actually mispronouncing anything. Use `jiwer` with a normalization transform (lowercase, strip punctuation, strip Arabic diacritics) before computing WER, or you will report inflated failure that isn't real and then have to explain a WER number you can't actually stand behind.

9. **Test sentences need to actually contain the hard cases, on purpose.** The brief specifically flags names and numbers as expected failure surfaces. If `test_sentences/{en,ar,hi}.txt` were bootstrapped generically, check by hand that each file has at least a few sentences with real names and real digit sequences (dates, phone-style numbers) in that language's own script — not romanized placeholders. Skipping this doesn't just weaken your eval, it quietly avoids the exact failure mode you're required to report, which reads as evasive even if unintentional.

### 1c. Worth adding, not required, cheap to add

10. A one-line **GPU memory budget note**: Whisper large-v3 + ECAPA-TDNN + a TTS model loaded simultaneously can be tight on a single T4 (16GB). If you hit an OOM, that's a legitimate documented failure mode, not something to route around and hide.

---

## 2. Hallucination audit — what to check and remove before you submit

Go through the repo once, specifically hunting for these:

- **Any placeholder number that looks like a real result.** Search the whole repo for stray digits that could be mistaken for a measured value — a comment like `# e.g. WER=8%` sitting near real code reads very differently once it's the only thing near a metric in a diff.
- **The bracketed placeholder in README's summary paragraph — confirm it's still bracketed.** You already know this; just don't let a "looks roughly right" filler number get typed in under time pressure to make the paragraph read cleanly. If real numbers aren't ready, the paragraph stays unfinished, full stop.
- **FAILURES.md checkboxes — don't check one without an actual sample file behind it.** A checked box with no corresponding `.wav` in `samples/generated/` is the single easiest thing for a reviewer to catch by just opening the folder.
- **Aggregate-only reporting.** If `build_report.py` is ever run against a partial or single-sample CSV and produces a clean-looking average, don't let that average stand in for the full per-utterance set in the README. State sample count next to every reported number (`WER: 6.2% (n=14)`), not just the number — an average with no denominator is unverifiable by construction.
- **Suspiciously uniform numbers across languages.** Covered in SPEC section 5 — worth a final manual glance at `build_report.py`'s output before submitting: if English/Arabic/Hindi land within a point of each other on MOS or WER, either that's a genuinely interesting finding worth calling out explicitly (unusual, and you should say so), or it's a sign something in the harness is broken (e.g., WER normalization silently failing and returning 0 for all languages) — check which before reporting either way.
- **Timestamps and commit hashes should not all be identical across every log entry** if the runs were spread over multiple days per the suggested 5-day window — if they are, that's fine as long as it's true (a single long session), but confirm it's true rather than assuming.

---

## 3. Run & deploy runbook

### Step 0 — Hardware decision (do this first, state it, don't change it mid-run)
Recommended: Google Colab (free T4, or Pro for an A100/longer sessions). Whatever you pick, name it exactly in the README header (e.g., "Colab, T4, 16GB VRAM") — an undisclosed hardware number is worthless per Section 6.

### Step 1 — Environment isolation (this is the step most likely to save you hours)
Don't install all three TTS libraries into one environment. Use **one virtual environment per model family**:
```bash
python -m venv envs/chatterbox && source envs/chatterbox/bin/activate
pip install -r requirements/chatterbox.txt

python -m venv envs/xtts && source envs/xtts/bin/activate
pip install -r requirements/xtts.txt

python -m venv envs/indic && source envs/indic/bin/activate
pip install -r requirements/indic.txt
```
`router.py` shells out to the right venv's Python via `subprocess` per language rather than importing all three into one process. This is more setup up front but avoids the single most likely early failure (a `torch`/`transformers` version fight between Coqui TTS and a Parler-TTS transformers build). If you've already got one shared `requirements.txt`, try it first — if `pip install` throws a resolver conflict, this is why, and this is the fix.

### Step 2 — Reference voice
Record ~15–20 seconds, quiet room, mono, 16kHz or 22kHz WAV (check what each TTS model's docs expect — don't assume they all want the same rate). Save to `samples/reference/`. This is your own voice, per the ground rules.

### Step 3 — Smoke test, English only
```bash
python -m eval.run_eval --lang en --reference-wav samples/reference/<file>.wav --limit 1
```
`--limit 1` (add this flag if it doesn't exist yet) so you're validating the full path — TTS call → gate → log — on one sentence before burning time on the full set. Confirm: a `.wav` lands in `samples/generated/`, a JSON log lands in `logs/`, and check by ear whether it sounds right before trusting the gate's verdict.

### Step 4 — Full English eval, then repeat per language
```bash
python -m eval.run_eval --lang en --reference-wav samples/reference/<file>.wav
python -m eval.run_eval --lang ar --reference-wav samples/reference/<file>.wav
python -m eval.run_eval --lang hi --reference-wav samples/reference/<file>.wav
```
Do these sequentially, not concurrently, if GPU memory is at all tight — load and release each model rather than holding all three plus two ASR models resident at once.

### Step 5 — MOS
Once real clips exist, send 5–8 clips per language to yourself + 2–3 others using the `mos_survey_template.csv` you build per gap 1a above. This is the one step that can't be automated away — budget real calendar time for it, not just compute time.

### Step 6 — Report
```bash
python -m eval.aggregate_mos
python -m eval.build_report
```
Read the output table once, specifically checking for the uniformity flag in section 2 above, before pasting anything into README.

### Step 7 — Fill FAILURES.md and README summary from real output only, then submit.

---

## 4. Edge cases worth deciding in advance, not discovering mid-run

- **A model fails to install entirely.** Don't silently drop that language. Document the install failure itself as a failure mode in FAILURES.md ("XTTS-v2 install failed on X, tried Y, fell back to Fish-Speech, results below") — an honest partial result beats a missing language.
- **TTS output is non-deterministic** (most of these models sample). Either fix seeds where the library exposes one, or run each sentence 2–3 times and report the spread — and say which you did. Don't let one lucky/unlucky sample stand in as "the" result without disclosing it's one draw among several possible.
- **Colab session disconnects mid-run.** Since logs are written per-sample (not just at the end), a disconnect mid-eval should leave partial real data behind rather than nothing — confirm this actually works by killing a run partway through once, on purpose, before it matters.
- **A language technically "passes" the gate on every metric except one you didn't think to threshold** (e.g., prosody sounds robotic despite good WER and cosine). The gate checks what's configured, not what's true — your own ear on the actual clips still matters; don't let a green checkmark substitute for actually listening.

---

## 5. Status of blocking items (1a)

- [x] MOS collection mechanism — `eval/generate_mos_survey.py`, `eval/mos_survey_template.csv`, `eval/aggregate_mos.py`
- [x] LIMITATIONS.md — open-source gaps, named
- [x] License check — documented in `LIMITATIONS.md`
- [x] Generation parameters logged per sample — `src/tts/*.py`, `src/router.py`, `eval/run_eval.py`
- [x] AI-disclosure line — `README.md`

## 6. Status of real-risk items (1b)

- [x] **Item 6 (dependency isolation)** — `requirements/{base,chatterbox,xtts,indic}.txt` split; `src/router.py` now dispatches to each model via `subprocess` against a per-model venv under `envs/` instead of importing all three TTS libraries into one process. Plumbing (arg-passing, meta-file handoff, error surfacing on subprocess failure) verified with a stub backend since no GPU/model libraries are installed in this environment — not run against the real TTS libraries yet.
- [x] **Item 7 (sample-rate mismatch)** — `src/similarity.py` now resamples both clips to 16kHz mono before ECAPA-TDNN embedding (`TARGET_SAMPLE_RATE`), instead of assuming a shared rate. Each `src/tts/*.py` backend also now reports its own native output sample rate (read from the model, not hardcoded) so the `.wav` file itself is written correctly, independent of the ECAPA-side fix.
- [x] **Item 8 (naive WER)** — `src/wer_normalize.py` added: lowercases, strips ASCII + Arabic/Hindi punctuation, strips Arabic diacritics (tashkil) before `jiwer.wer()` runs, verified against real sentences from `eval/test_sentences/{ar,hi}.txt`. Wired into `eval/run_eval.py`.
- [x] **Item 9 (hard-case test sentences)** — turned out to already be satisfied on inspection: `eval/test_sentences/{en,ar,hi}.txt` each contain real names (Priya Sharma / أحمد / अनिरुद्ध) and digit sequences (phone numbers, dates, ticket numbers) in native script. No code change needed.

## 7. Still open (1c — cheap, optional)

- [ ] GPU memory budget note — documented as an untested risk in `LIMITATIONS.md` section 2, not yet exercised against real concurrent model loads (no GPU available in this environment).

## 8. Real failure encountered on first actual run (2026-07-15, Colab, Python 3.12)

`pip install -r requirements.txt` failed entirely: `ERROR: No matching
distribution found for TTS`. The original Coqui `TTS` PyPI package is
unmaintained since 2023 and does not resolve on Python 3.10+ -- Colab's
default runtime is Python 3.12. Because pip's modern resolver plans the
full install before installing anything, this single failure blocked
*every* package in the file, including unrelated ones like `jiwer`, which
then surfaced as a separate, more confusing `ModuleNotFoundError` at
eval-run time.

**Fix applied:** swapped `TTS` for `coqui-tts` in `requirements.txt` and
`requirements/xtts.txt` -- the actively-maintained community fork, same
`from TTS.api import TTS` import surface, no source code changes needed.
Not yet re-verified against a real Colab run past this point (next TTS
library in the list to potentially hit a similar issue: `chatterbox-tts`
or `parler-tts`, unconfirmed either way).
