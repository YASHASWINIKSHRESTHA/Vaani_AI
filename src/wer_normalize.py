"""Text normalization applied before computing round-trip WER.

Without this, raw ASR-transcript-vs-input-text comparison gets dinged for
punctuation, casing, and (Arabic) diacritics that have nothing to do with
whether the TTS actually mispronounced anything -- overstating failure on
Arabic and Hindi specifically (GAPS.md section 1b, item 8).

All non-ASCII code points below are written as \\uXXXX escapes rather than
literal characters, so the intended range is unambiguous regardless of
editor/terminal encoding.
"""
import re
import string

# Arabic combining diacritics (tashkil/harakat): Quranic annotation marks
# (U+0610-U+061A), fatha/damma/kasra/sukun/shadda/tanwin and friends
# (U+064B-U+065F), superscript alef (U+0670), further Quranic marks
# (U+06D6-U+06ED).
_ARABIC_DIACRITICS = re.compile(
    "[ؐ-ًؚ-ٰٟۖ-ۭ]"
)

# Non-ASCII punctuation used in Arabic and Hindi text that Python's
# string.punctuation (ASCII-only) doesn't cover: Arabic comma (U+060C),
# semicolon (U+061B), question mark (U+061F); guillemets (U+00AB/00BB);
# curly quotes (U+201C/201D/2018/2019); Devanagari danda/double danda
# (U+0964/0965); ellipsis (U+2026); em/en dash (U+2014/2013).
_EXTRA_PUNCTUATION = (
    "،؛؟"
    "«»"
    "“”‘’"
    "।॥"
    "…—–"
)


def normalize_for_wer(text: str, language: str) -> str:
    """Lowercases, strips punctuation (including Arabic/Hindi punctuation),
    and strips Arabic diacritics, then collapses whitespace. Apply to both
    the reference text and the ASR transcript before diffing.
    """
    if language == "ar":
        text = _ARABIC_DIACRITICS.sub("", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation and ch not in _EXTRA_PUNCTUATION)
    text = re.sub(r"\s+", " ", text).strip()
    return text
