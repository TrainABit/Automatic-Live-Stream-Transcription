"""One text normaliser for speech: NFC and case folding, German folding optional.

STT providers do not agree on a Unicode form. "überlege" can arrive with a
precomposed ``ü`` (U+00FC) or as ``u`` + combining diaeresis (U+0308). Compared
naively the two spellings differ, and a decomposed mark is not a word
character, so a tokeniser splits the word in two. Every keyword list and every
token comparison is therefore written against the single form produced here.

The default is language neutral: NFC, ``str.casefold``, punctuation collapsed
to single spaces. ``fold_umlauts=True`` additionally maps ``ä ö ü ß`` to
``ae oe ue ss`` so that "Straße" matches "strasse" and "Müller" matches
"Mueller"; it is opt-in because it is wrong for most other languages.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["fold_german", "nfc", "norm_text", "norm_token", "norm_words"]

_GERMAN_FOLD = str.maketrans({"ß": "ss", "ä": "ae", "ö": "oe", "ü": "ue"})
_NON_WORD = re.compile(r"[^\w]+")


def nfc(text: str | None) -> str:
    """``text`` in Unicode NFC: composed letters, one code point per umlaut."""
    return unicodedata.normalize("NFC", text or "")


def fold_german(text: str | None) -> str:
    """NFC, lower case, ``ä ö ü ß`` -> ``ae oe ue ss``. Punctuation is kept."""
    return nfc(text).lower().translate(_GERMAN_FOLD)


def norm_text(text: str | None, *, fold_umlauts: bool = False) -> str:
    """Case-fold, then collapse every run of non-word characters to one space."""
    base = fold_german(text) if fold_umlauts else nfc(nfc(text).casefold())
    return " ".join(_NON_WORD.sub(" ", base).split())


def norm_words(text: str | None, *, fold_umlauts: bool = False) -> list[str]:
    """The words of :func:`norm_text`."""
    return norm_text(text, fold_umlauts=fold_umlauts).split()


def norm_token(text: str | None, *, fold_umlauts: bool = False) -> str:
    """One comparable token: :func:`norm_text` with the spaces removed."""
    return "".join(norm_words(text, fold_umlauts=fold_umlauts))
