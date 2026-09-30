from __future__ import annotations

import unicodedata

import pytest

from livestream_transcriber.textnorm import fold_german, nfc, norm_text, norm_token, norm_words

PRECOMPOSED = "überlege"  # ü as one code point
DECOMPOSED = "überlege"  # u + combining diaeresis


def test_both_unicode_forms_normalise_identically():
    assert PRECOMPOSED != DECOMPOSED
    assert norm_text(PRECOMPOSED) == norm_text(DECOMPOSED)
    assert norm_words(DECOMPOSED) == ["überlege"], "the word must not split at the mark"


def test_nfc_composes_and_tolerates_none():
    assert unicodedata.is_normalized("NFC", nfc(DECOMPOSED))
    assert nfc(None) == ""


def test_default_normalisation_is_language_neutral():
    assert norm_text("Willkommen zum STREAM!") == "willkommen zum stream"
    assert norm_text("Straße") == "strasse", "casefold maps ß to ss on its own"
    assert norm_text("Müller") == "müller", "umlauts are kept unless folding is requested"
    assert norm_text("İstanbul") != "istanbul"  # no accidental locale-specific mangling


def test_optional_german_folding():
    assert norm_text("Müller Größe", fold_umlauts=True) == "mueller groesse"
    assert fold_german("Ärger, Öl & Übung!") == "aerger, oel & uebung!"
    assert norm_words("Schön, oder?", fold_umlauts=True) == ["schoen", "oder"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello,   world!! ", "hello world"),
        ("a\tb\nc", "a b c"),
        ("well-known", "well known"),
        ("", ""),
        (None, ""),
        ("!!!", ""),
    ],
)
def test_punctuation_and_whitespace_collapse(raw: str | None, expected: str):
    assert norm_text(raw) == expected


def test_token_joins_words():
    assert norm_token("Good Morning!") == "goodmorning"
    assert norm_token(None) == ""
