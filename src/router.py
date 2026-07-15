"""Propose layer: dispatches (text, language, reference_voice) to the right
TTS model.

Each backend runs as a subprocess in its own venv (envs/<name>/) rather than
being imported into this process. Coqui TTS (XTTS-v2), Chatterbox, and
transformers-based Parler-TTS commonly pin incompatible torch/transformers
versions against each other -- importing all three here is the most likely
first thing to break (GAPS.md section 1b, item 6). This is more setup than
a single shared environment, but avoids that version fight entirely.
"""
import json
import subprocess
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "language_models.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "envs"

# model_key -> (venv name under envs/, module to run via `python -m`)
_BACKENDS = {
    "chatterbox": ("chatterbox", "src.tts.chatterbox"),
    "xtts_v2": ("xtts", "src.tts.xtts"),
    "indic_parler_tts": ("indic", "src.tts.indic_parler"),
}


def load_language_map() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _venv_python(env_name: str) -> str:
    """Resolves the interpreter for envs/<env_name>, created per README
    "Environment setup" (`python -m venv envs/<env_name>` then
    `pip install -r requirements/<env_name>.txt`)."""
    for candidate in (
        ENVS_DIR / env_name / "Scripts" / "python.exe",  # Windows
        ENVS_DIR / env_name / "bin" / "python",  # POSIX
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"No venv found at {ENVS_DIR / env_name}. Create it first:\n"
        f"  python -m venv {ENVS_DIR / env_name}\n"
        f"  {ENVS_DIR / env_name} activate, then: "
        f"pip install -r requirements/{env_name}.txt"
    )


def synthesize(
    text: str,
    language: str,
    reference_wav: str,
    out_wav: str,
    use_fallback: bool = False,
) -> dict:
    """Routes a synthesis request to the configured primary (or fallback)
    model for `language`, running it in that model's isolated venv.

    Writes the generated clip to `out_wav` (at the model's native sample
    rate) and returns {"params": ..., "sample_rate": ...} so callers can log
    the exact generation settings alongside model identity (SPEC.md
    section 5, reproducibility requirement).
    """
    language_map = load_language_map()
    if language not in language_map:
        raise ValueError(f"No model configured for language '{language}'")

    model_key = language_map[language]["fallback" if use_fallback else "primary"]
    if model_key not in _BACKENDS:
        raise ValueError(f"No backend wired up for model '{model_key}'")

    env_name, module = _BACKENDS[model_key]
    python = _venv_python(env_name)

    cmd = [
        python, "-m", module,
        "--text", text,
        "--reference-wav", reference_wav,
        "--out-wav", out_wav,
    ]
    if model_key == "xtts_v2":
        cmd += ["--language", language]

    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TTS subprocess failed for model '{model_key}' (language '{language}'):\n"
            f"{proc.stderr}"
        )

    meta_path = Path(out_wav + ".meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
