"""
strip_emojis — the ONE shared emoji/emoticon scrubber for every clinical
response the system produces.

Why a deterministic scrubber, not just a prompt instruction
    A system-prompt instruction ("never use emojis") reduces how often the
    model reaches for one, but it is not a guarantee: the underlying LLM can
    still emit one, and two call sites in this codebase inject emojis
    directly in Python literals (the canned emergency message, the follow-up
    prompt suffix) independent of what the model does at all. Prompt-only
    enforcement would leave both of those untouched. This function is applied
    to every clinical response text regardless of its source, so the
    guarantee holds even if a model update, a new canned string, or a new
    specialty forgets the prompt instruction.

What it removes
    * Unicode emoji ranges (pictographs, emoticons, dingbats, transport and
      map symbols, flags, the misc-symbols block that carries U+26A0 WARNING
      SIGN, supplemental symbols and pictographs, chess/alchemical blocks).
    * Emoji presentation/joining marks left dangling after the above
      (variation selectors U+FE00-FE0F, zero-width joiner, combining
      enclosing keycap) so a multi-codepoint sequence never leaves an orphan
      combining mark behind.
    * A short, deliberately narrow list of ASCII emoticons ( :) :-) :( :-(
      ;) :D :-D :P ), matched only at a word/whitespace boundary so clinical
      shorthand that merely contains a colon (a time "3:30", a ratio "1:1",
      a dose ratio) is never touched.

What it deliberately leaves alone
    Plain arrows (-> U+2192, <- U+2190) and the Unicode arrow block
    (U+2190-21FF) are NOT emoji and are used throughout this codebase's own
    comments/log lines; they are excluded so "preserve formatting" holds.
    Markdown (**bold**, bullet markers, newlines) is untouched — only the
    emoji/emoticon characters themselves are removed, and only the single
    extra space an inline removal can leave behind is collapsed back to one.
"""

from __future__ import annotations

import re

# Standard Unicode emoji blocks. Deliberately excludes the plain Arrows block
# (U+2190-21FF) and Supplemental Arrows-C (U+1F800-1F8FF), neither of which
# this codebase or a clinical response should have its arrows stripped from.
_EMOJI_RANGES = (
    "\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons (the classic smiley-face block)
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flag letters)
    "☀-⛿"  # misc symbols (includes U+26A0 WARNING SIGN)
    "✀-➿"  # dingbats (checkmarks, crosses, scissors, ...)
    "⬀-⯿"  # misc symbols and arrows (star, large circle, ...)
    "⌀-⏿"  # misc technical (watch/hourglass/media-control glyphs)
)
_EMOJI_MODIFIERS = "︀-️‍⃣"  # variation selectors, ZWJ, keycap

_EMOJI_RE = re.compile(f"[{_EMOJI_RANGES}{_EMOJI_MODIFIERS}]")

# Narrow, boundary-anchored ASCII emoticon list. Longer alternatives first so
# ":-)" matches before the bare ":)" pattern could partially consume it.
_ASCII_EMOTICON_RE = re.compile(
    r"(?<![:\w])(?::-?\)|:-?\(|:-?D|;-?\)|:-?P)(?![:\w])",
    re.IGNORECASE,
)

# A run of 2+ ASCII spaces (never tabs/newlines) — the inline case an emoji
# removal creates when the glyph sat BETWEEN two words ("great 😊 news").
_INLINE_DOUBLE_SPACE_RE = re.compile(r"(?<=\S)  +(?=\S)")
# Exactly one leading space followed by non-space — the artifact a removed
# leading glyph leaves ("🚨 Medical" -> " Medical"). Deliberately requires a
# SINGLE space: genuine markdown indentation (nested bullets) uses two or
# more, which this does not match, so real indentation survives untouched.
_SINGLE_LEADING_SPACE_RE = re.compile(r"^ (?=\S)")


def strip_emojis(text: str | None) -> str | None:
    """
    Remove emoji and ASCII emoticons from ``text``, preserving everything
    else — wording, punctuation, markdown, and line structure.

    ``None`` and ``""`` pass through unchanged so every call site can apply
    this unconditionally without an extra guard. Never raises: this runs on
    the hot clinical-response path and a sanitizer bug must never surface as
    a broken turn, so any unexpected input type is returned as-is.
    """
    if not text:
        return text
    if not isinstance(text, str):
        return text

    out = _EMOJI_RE.sub("", text)
    out = _ASCII_EMOTICON_RE.sub("", out)

    # Clean up only the two whitespace artifacts a removal can leave, line by
    # line, never touching newlines or genuine indentation:
    #   - "great 😊 news"  -> "great  news"  -> "great news"   (inline)
    #   - "🚨 Medical ..."  -> " Medical ..." -> "Medical ..."  (leading)
    #   - "... alert 🚨"    -> "... alert "   -> "... alert"    (trailing)
    lines = out.split("\n")
    lines = [
        _SINGLE_LEADING_SPACE_RE.sub("", _INLINE_DOUBLE_SPACE_RE.sub(" ", line)).rstrip(" \t")
        for line in lines
    ]
    return "\n".join(lines)


def sanitize_value(value):
    """
    Recursively apply :func:`strip_emojis` to every string inside ``value``.

    Generic over the shapes the response pipeline actually produces, so a
    new block/SOAP field needs no new sanitizer code: a plain string is
    scrubbed directly; a list or tuple is scrubbed element-by-element; a dict
    is scrubbed value-by-value (used for the SOAP note's plain dict); a
    pydantic ``BaseModel`` (a block's ``.data``, or a nested item like
    ``Condition``/``OtcMedication``) is rebuilt via ``model_copy`` with its
    string-bearing fields scrubbed. Anything else (bool, float, an enum, a
    frozen type the model carries unchanged) is returned as-is.

    Never raises: falls back to returning ``value`` unchanged if the walk
    hits something it doesn't recognise, so a sanitizer bug can never turn
    into a broken turn.
    """
    try:
        if isinstance(value, str):
            return strip_emojis(value)
        if isinstance(value, (list, tuple)):
            cleaned = [sanitize_value(v) for v in value]
            return type(value)(cleaned)
        if isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            updates = {k: sanitize_value(v) for k, v in value.__dict__.items()}
            return value.model_copy(update=updates)
        return value
    except Exception:  # noqa: BLE001 - sanitizing must never break a turn
        return value


__all__ = ["sanitize_value", "strip_emojis"]
