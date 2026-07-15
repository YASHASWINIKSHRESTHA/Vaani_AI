# Limitations & What's Missing

This is separate from [FAILURES.md](FAILURES.md). FAILURES.md documents specific
sentences that broke during the eval run. This file documents structural gaps
in the pipeline and in the open-source ecosystem it's built on — things no
amount of threshold-tuning would fix.

## 1. Licensing — not all "open source" here is deployable

| Model | License | Commercial use? |
|---|---|---|
| Chatterbox (Resemble AI) | MIT | Yes |
| AI4Bharat Indic Parler-TTS | Apache 2.0 | Yes |
| XTTS-v2 (Coqui) | Coqui Public Model License (CPML) | **No — non-commercial only** |
| Meta MMS-TTS (Arabic fallback-of-fallback, SPEC section 4) | CC-BY-NC | **No — non-commercial only** |

XTTS-v2 is used here as the Arabic primary model because it is, at present,
the strongest openly-available zero-shot cloner with explicit MSA support.
That is a valid choice for a take-home evaluation. It is **not** a valid
choice for a shipped product without buying a commercial CPML license.

The deeper finding, not just a caveat on one model: **commercially-licensed
open-source Arabic TTS is genuinely thin right now.** Every credible
zero-shot option surveyed for this pipeline (XTTS-v2, MMS-TTS) carries a
non-commercial license. Any real deployment targeting Arabic needs either a
commercial license negotiation, a closed API (which then reopens the
"is this actually open source" question the brief cares about), or a
lower-quality fully-permissive fallback. This is a gap in the ecosystem, not
in this implementation.

## 2. What's still missing from this pipeline

- **No prosody/naturalness metric beyond MOS.** WER and speaker cosine catch
  intelligibility and voice-identity failures but not "technically correct,
  sounds robotic" — the gate can go green while a clip still sounds wrong.
  MOS is the only signal for this, and it's human-bottlenecked (see below).
- **MOS is small-panel by construction** (2–3 listeners, per the brief's own
  allowance) — not a statistically powered listener study. Treat MOS numbers
  as directional, not precise, and say so next to the number.
- **Round-trip WER is a proxy for intelligibility, not a direct measure of
  naturalness or correctness of prosody.** A clip can have 0% WER and still
  sound bad; conversely, WER can be inflated by ASR normalization edge cases
  unrelated to the TTS.
- **No streaming latency measurement path yet** — the harness measures batch
  synthesis; the brief's `<500ms first-audio streaming` target is not
  currently exercised by anything in `eval/`. If none of the three models are
  run in a streaming mode, report the batch number honestly and say
  streaming was not measured, rather than implying a target was hit.
- **Single reference speaker.** The pipeline is only validated against
  whatever one voice is recorded into `samples/reference/`. Robustness across
  different speaker timbres, accents, or recording conditions is untested.
- **No code-switched sentence coverage guaranteed.** SPEC section 5 lists
  code-switching as a bonus test case, not a requirement — confirm whether
  `eval/test_sentences/*.txt` actually includes any before claiming the
  pipeline handles it.
- **GPU memory budget is untested.** Whisper large-v3 + ECAPA-TDNN + a TTS
  model loaded simultaneously on a single T4 (16GB) may not fit. If an OOM
  is hit during the real run, treat it as a documented failure mode in
  FAILURES.md, not something to silently route around.

## 3. If this were taken further

- Swap XTTS-v2 for a commercially-licensed Arabic model as soon as one
  reaches comparable quality (worth re-checking periodically — this is a
  fast-moving space).
- Add a streaming-mode benchmark for whichever primary models support it,
  to actually answer the brief's latency target instead of leaving it
  unmeasured.
- Expand the MOS panel beyond 2–3 listeners if the submission timeline
  allows, and report inter-rater spread, not just the mean.
