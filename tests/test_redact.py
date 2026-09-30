"""Nothing that reaches a log sink may carry a secret.

Debug logs of a naive implementation contain signed media URLs, playback
signatures and the capturing machine's IP address. These tests are the
regression fence.
"""

from __future__ import annotations

import json
import logging

import pytest

from livestream_transcriber.logging_setup import JsonlFormatter, setup_logging
from livestream_transcriber.redact import (
    REDACTED,
    RedactingFilter,
    describe_command,
    is_sensitive_url,
    redact_argv,
    redact_mapping,
    redact_text,
    redact_url,
)

# A realistically shaped resolved rendition: signature, expiry, edge node and
# the capturing machine's public IPv6 address all in one string.
SIGNED = (
    "https://rr3---sn-edge0001.googlevideo.com/videoplayback"
    "?expire=1757443200&ei=0aBc-XYZ&ip=2001%3Adb8%3A5810%3A1%3A9c11%3A1"
    "&ipbits=0&id=o-FAKEid&mn=sn-edge0001&mm=31&source=youtube"
    "&sig=FAKEsig0123456789abc&lsig=FAKElsig0123"
    "&mime=audio%2Fmp4&itag=140&cpn=7HqL2mQ"
)
BENIGN = "https://www.youtube.com/watch?v=demoVideo01"
# Synthetic credentials. They need the shape of the real thing for the redaction patterns to
# fire, so they are assembled from parts and never appear as key-shaped literals.
TELEGRAM_TOKEN = "123456789" + ":" + "AAH1a-ZzQqRrSsTtUuVvWwXxYyZz012345678"
OPENAI_KEY = "sk" + "-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
CLOUD_ACCESS_ID = "AK" + "IAIOSFODNN7EXAMPLE"


# --------------------------------------------------------------------- URLs


def test_signed_media_url_is_scrubbed():
    out = redact_url(SIGNED)
    assert "googlevideo.com" in out, "the host suffix is useful and not secret"
    assert REDACTED in out


@pytest.mark.parametrize(
    "leak",
    [
        "FAKEsig0123456789abc",  # sig
        "FAKElsig0123",  # lsig
        "2001",  # IPv6 address of the capturing machine
        "db8",
        "sn-edge0001",  # edge node / network location
        "1757443200",  # expire
        "7HqL2mQ",  # client identifier
    ],
)
def test_no_signed_url_component_survives(leak: str):
    assert leak not in redact_url(SIGNED)
    assert leak not in redact_text(f"ffmpeg failed on {SIGNED} after 3 tries")


def test_signature_and_ip_gone_from_argv():
    argv = ["ffmpeg", "-hide_banner", "-i", SIGNED, "-f", "s16le", "pipe:1"]
    joined = " ".join(redact_argv(argv))
    assert "sig=" not in joined
    assert "ip=" not in joined
    assert "2001" not in joined


def test_benign_url_survives_intact():
    assert redact_url(BENIGN) == BENIGN


def test_userinfo_credentials_are_scrubbed():
    assert "hunter2" not in redact_url("https://admin:hunter2@example.com/feed")


def test_sensitive_param_on_a_benign_host_is_dropped():
    out = redact_url("https://example.com/a?page=2&access_token=your-token-value")  # gitleaks:allow
    assert "your-token-value" not in out
    assert "page=2" in out, "non-sensitive params still aid debugging"


def test_rtmps_urls_are_recognised_in_text():
    text = "publishing to rtmps://user:hunter2@ingest.example.com/live now"
    assert "hunter2" not in redact_text(text)


def test_is_sensitive_url():
    assert is_sensitive_url(SIGNED)
    assert is_sensitive_url("https://cdn.example.com/x?X-Amz-Signature=deadbeef")
    assert not is_sensitive_url(BENIGN)


# -------------------------------------------------------------- credentials


def test_telegram_token_is_scrubbed():
    msg = f"POST https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage -> 401"
    out = redact_text(msg)
    assert TELEGRAM_TOKEN not in out
    assert "AAH1a-Zz" not in out
    assert "401" in out, "the status code is the useful part"


def test_bare_telegram_token_is_scrubbed():
    assert TELEGRAM_TOKEN not in redact_text(f"configured token {TELEGRAM_TOKEN} ok")


def test_api_key_is_scrubbed():
    out = redact_text(f"Incorrect API key provided: {OPENAI_KEY}")
    assert OPENAI_KEY not in out
    assert REDACTED in out


def test_bearer_and_header_credentials_are_scrubbed():
    assert "eyJhbGciOi" not in redact_text("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
    assert "s3cr3t-value-x" not in redact_text("x-api-key: s3cr3t-value-x")


def test_aws_key_is_scrubbed():
    assert CLOUD_ACCESS_ID not in redact_text(f"using {CLOUD_ACCESS_ID} now")


@pytest.mark.parametrize("value", ["speech", "en", "chunk=12", "queue full", "18x"])
def test_short_operational_strings_are_untouched(value: str):
    # The per-chunk hot path must not be mangled.
    assert redact_text(value) == value


def test_redact_mapping_recurses():
    out = redact_mapping({"n": 7, "u": SIGNED, "nested": {"argv": [SIGNED]}})
    assert out["n"] == 7
    assert "sig=" not in json.dumps(out)


# ------------------------------------------------------------------ command


def test_describe_command_keeps_flags_and_hides_urls():
    argv = [
        "ffmpeg", "-hide_banner", "-reconnect", "1",
        "-i", SIGNED, "-i", SIGNED,
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
    ]  # fmt: skip
    info = describe_command(argv)
    assert info["binary"] == "ffmpeg"
    assert info["input_count"] == 2
    # Every debugging flag survives...
    for flag in ("-reconnect", "-hide_banner", "s16le", "16000"):
        assert flag in info["argv"]
    # ...and no secret does.
    blob = json.dumps(info)
    assert "sig=" not in blob
    assert "2001" not in blob
    assert "sn-edge0001" not in blob


def test_describe_command_of_nothing():
    assert describe_command([])["binary"] == ""


# ------------------------------------------------------- the logging sinks


def _record(msg: str, **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, msg, (), None)
    rec.__dict__.update(extra)
    return rec


def test_logging_filter_scrubs_message_and_extras():
    rec = _record("starting ffmpeg", argv=f"ffmpeg -i {SIGNED}", token=TELEGRAM_TOKEN)
    assert RedactingFilter().filter(rec) is True
    blob = JsonlFormatter().format(rec)
    assert "sig=" not in blob
    assert "2001" not in blob
    assert TELEGRAM_TOKEN not in blob


def test_logging_filter_scrubs_printf_args():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "ffmpeg: %s", (SIGNED,), None)
    RedactingFilter().filter(rec)
    assert "sig=" not in rec.getMessage()


def test_logging_filter_scrubs_mapping_args():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "%(u)s", ({"u": SIGNED},), None)
    RedactingFilter().filter(rec)
    assert "sig=" not in rec.getMessage()


def test_setup_logging_installs_the_filter_on_every_handler(tmp_path):
    log_file = tmp_path / "debug.jsonl"
    setup_logging("DEBUG", json_path=log_file, color=False)
    try:
        logging.getLogger("livestream_transcriber.test").debug(
            "starting ffmpeg", extra={"argv": f"ffmpeg -i {SIGNED}"}
        )
        for handler in logging.getLogger().handlers:
            handler.flush()
        written = log_file.read_text()
    finally:
        setup_logging("INFO", color=False)
    assert "starting ffmpeg" in written
    for leak in ("sig=FAKE", "2001", "sn-edge0001", "expire=1757443200"):
        assert leak not in written
