"""Single source of truth for scrubbing secrets out of anything that is logged.

Every outbound text path (log lines, error strings, ffmpeg argv, recorder
manifests) has to obey the same rule, or the protection is theatre. A resolved
media URL from a video platform carries, in its query string alone:

* ``sig`` / ``lsig`` / ``sparams``: the playback signature;
* ``expire``: when it dies, i.e. when the capture started;
* ``ip`` / ``ipbits``: **the public IP of the capturing machine**;
* ``mn`` / ``mm`` / ``ei`` / ``cpn``: edge node and client identifiers.

So: never log a URL that has been through the resolver, never log an argv that
might contain one, and never log a bearer token, chat-bot token or API key.
Use :func:`redact_text` for free-form strings, :func:`redact_url` for a value
known to be a URL, and :func:`redact_argv` for a subprocess command.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

__all__ = [
    "REDACTED",
    "RedactingFilter",
    "describe_command",
    "is_sensitive_url",
    "redact_argv",
    "redact_mapping",
    "redact_text",
    "redact_url",
]

REDACTED = "<redacted>"

# Query parameters that are, or leak, a secret. Matched case-insensitively.
_SENSITIVE_PARAMS = frozenset(
    {
        # signature / authorisation
        "sig", "signature", "lsig", "msig", "hmac", "sparams", "pcm2",
        "key", "api_key", "apikey", "auth", "authorization", "access_token",
        "token", "id_token", "refresh_token", "secret", "client_secret",
        "password", "passwd", "pwd", "session", "sid", "sessionid",
        "x-amz-signature", "x-amz-credential", "x-amz-security-token",
        "x-goog-signature", "x-goog-credential", "policy",
        # identity / network location of the capturing machine
        "ip", "ipbits", "ipv4", "ipv6", "cpn", "ei", "mn", "mm", "ms", "mv",
        "expire", "gir", "initcwndbps", "pl", "requiressl", "spc", "vprv",
    }
)  # fmt: skip

# Hosts whose URLs are signed media by construction: redacted even with no query.
_SIGNED_HOSTS = (
    "googlevideo.com",
    "video.google.com",
    "ytimg.com",
    "akamaized.net",
    "cloudfront.net",
    "llnwd.net",
    "fbcdn.net",
    "twimg.com",
    "cloudflarestream.com",
)

_URL_RE = re.compile(r"\b(?:https?|rtmps?|wss?)://[^\s'\"<>\\|]+", re.IGNORECASE)

# Credentials that can appear in free text (exception messages, stderr, argv).
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Telegram bot token, both bare and inside an api.telegram.org path.
    (re.compile(r"\bbot\d{6,12}:[A-Za-z0-9_-]{20,}", re.IGNORECASE), f"bot{REDACTED}"),
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}"), REDACTED),
    # OpenAI / OpenRouter / Anthropic style keys.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), REDACTED),
    # AWS access key ids and generic bearer/basic credentials.
    (re.compile(r"\b(?:AK|AS)IA[0-9A-Z]{16}\b"), REDACTED),
    (
        re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
        f"Bearer {REDACTED}",
    ),
    (
        re.compile(
            r"\b(authorization|x-api-key|api[-_]?key|access[-_]?token|secret)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{8,}['\"]?",
            re.IGNORECASE,
        ),
        rf"\1={REDACTED}",
    ),
)

# Cheap gate: the shortest thing any pattern above can match is 15 characters
# ("Bearer " + 8), so short per-chunk debug extras ("speech", counts) skip the
# regex work entirely. Deliberately conservative: correctness first.
_MIN_SECRET_LEN = 12


def _signed_suffix(host: str) -> str | None:
    """The entry of ``_SIGNED_HOSTS`` that ``host`` belongs to, if any."""
    return next((h for h in _SIGNED_HOSTS if host == h or host.endswith("." + h)), None)


def is_sensitive_url(url: str) -> bool:
    """True when *url* is a signed media URL or carries a secret parameter."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return True  # unparseable: assume the worst
    if _signed_suffix((parts.hostname or "").lower()) or parts.username or parts.password:
        return True
    return any(
        name.lower() in _SENSITIVE_PARAMS
        for name, _value in parse_qsl(parts.query, keep_blank_values=True)
    )


def redact_url(url: str) -> str:
    """Return *url* with every secret removed, keeping only what aids debugging.

    A signed rendition collapses to its host suffix and path shape::

        https://rr3---sn-4g5e6.googlevideo.com/videoplayback?sig=AB..&ip=2a02::1
        -> https://<redacted>.googlevideo.com/videoplayback?<redacted>

    A benign URL keeps its structure and loses only sensitive parameters, so
    ``https://www.youtube.com/watch?v=demoVideo01`` survives intact.
    """
    if not isinstance(url, str) or "://" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return REDACTED

    host = (parts.hostname or "").lower()
    suffix = _signed_suffix(host)

    if suffix:
        # Keep the registrable suffix; the leading `rr3---sn-...` label
        # identifies the edge node that served this machine.
        netloc = suffix if host == suffix else f"{REDACTED}.{suffix}"
    elif parts.username or parts.password:
        netloc = f"{REDACTED}@{host}" + (f":{parts.port}" if parts.port else "")
    else:
        netloc = parts.netloc

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        kept = [(n, v) for n, v in pairs if n.lower() not in _SENSITIVE_PARAMS]
        if suffix or len(kept) != len(pairs):
            if suffix or not kept:
                query = REDACTED
            else:
                query = "&".join(f"{n}={v}" for n, v in kept) + f"&{REDACTED}"

    fragment = REDACTED if parts.fragment and suffix else parts.fragment
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def redact_text(text: str) -> str:
    """Scrub URLs and credentials out of an arbitrary string."""
    if not isinstance(text, str) or len(text) < _MIN_SECRET_LEN:
        return text
    out = _URL_RE.sub(lambda m: redact_url(m.group(0)), text)
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Per-argument redaction of a subprocess command line."""
    return [redact_text(str(a)) for a in argv]


def describe_command(argv: Sequence[str]) -> dict[str, Any]:
    """A log-safe summary of a subprocess command.

    Every flag is preserved, because those are what you need when ffmpeg
    misbehaves; only URL-shaped arguments collapse.
    """
    safe = redact_argv(argv)
    inputs = [safe[i + 1] for i, a in enumerate(safe) if a == "-i" and i + 1 < len(safe)]
    return {
        "binary": safe[0] if safe else "",
        "argv": " ".join(safe),
        "inputs": inputs,
        "input_count": len(inputs),
    }


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Redact the string leaves of a mapping, recursing into containers."""
    return {k: _redact_value(v) for k, v in data.items()}


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    return value


# Record attributes that are ours by construction and never secret; skipping
# them keeps the per-chunk debug path cheap.
_RECORD_SKIP = frozenset(
    {"name", "levelname", "pathname", "filename", "module", "funcName", "processName", "threadName"}
)


class RedactingFilter:
    """Last line of defence: scrub every record before any handler formats it.

    Call sites are expected to redact deliberately; this filter exists so that
    a future ``log.debug("url %s", signed_url)`` cannot leak, and so that third
    party loggers (yt-dlp, urllib) are covered too.
    """

    def filter(self, record: Any) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_mapping(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact_value(a) for a in record.args)
        for key, value in list(record.__dict__.items()):
            if key in _RECORD_SKIP or not isinstance(value, (str, dict, list, tuple)):
                continue
            record.__dict__[key] = _redact_value(value)
        return True
