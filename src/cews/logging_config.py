"""Logging setup for CEWS, including redaction of API keys and other secrets.

``configure_logging`` loads ``config/logging.yaml``, applies the configured level, points the
rotating file handler at ``<project>/logs/cews.log`` (or a caller-supplied directory), and
installs :class:`RedactingFilter` on every handler so secrets never reach a log file or the
console.

Limitation: the filter redacts log *messages*. Secrets embedded in exception tracebacks are
not rewritten, so never put a secret in an exception message.
"""

from __future__ import annotations

import logging
import logging.config
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from cews.constants import CONFIG_DIR
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

REDACTED = "***"
_KEY_VALUE = re.compile(r"""(?ix)
    (?P<key>api[_-]?key|apikey|access[_-]?token|token|secret|password|passwd|consumer[_-]?secret)
    (?P<sep>["']?\s*[:=]\s*["']?)
    (?P<value>[^\s"'&,;]+)
    """)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with known secrets and common credential patterns masked.

    Args:
        text: The text to clean.
        secrets: Exact secret values to mask wherever they appear (empty values are ignored).
    """
    result = text
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        result = result.replace(secret, REDACTED)
    result = _URL_CREDENTIALS.sub(lambda m: f"{m.group('scheme')}{REDACTED}@", result)
    result = _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", result)
    return _KEY_VALUE.sub(lambda m: f"{m.group('key')}{m.group('sep')}{REDACTED}", result)


class RedactingFilter(logging.Filter):
    """Logging filter that masks secrets in the formatted message of each record."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets: list[str] = [s for s in secrets if s]

    def update_secrets(self, secrets: Iterable[str]) -> None:
        """Replace the set of exact secret values to mask."""
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record message in place; always keep the record."""
        message = record.getMessage()
        cleaned = redact_secrets(message, self._secrets)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


def _load_logging_config(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read logging configuration {path}: {exc}") from exc
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"{path} is not a valid logging dictConfig mapping")
    return data


def configure_logging(
    settings: Settings,
    config_path: Path | None = None,
    log_dir: Path | None = None,
    enable_file: bool = True,
) -> None:
    """Configure logging from ``config/logging.yaml`` and the given settings.

    Safe to call more than once: existing redaction filters are updated, not duplicated.

    Args:
        settings: Provides the log level, project root, and secrets to redact.
        config_path: Alternative logging YAML (defaults to ``config/logging.yaml``).
        log_dir: Directory for ``cews.log`` (defaults to ``<project_root>/logs``).
        enable_file: When false, only console logging is configured (useful in tests).

    Raises:
        ValueError: if the logging configuration cannot be read.
    """
    path = config_path or (settings.project_root / "config" / "logging.yaml")
    if not path.is_file():
        path = CONFIG_DIR / "logging.yaml"
    config = _load_logging_config(path)

    level = settings.log_level
    for handler in config.get("handlers", {}).values():
        handler["level"] = level
    config.setdefault("loggers", {}).setdefault("cews", {})["level"] = level

    if enable_file and "file" in config.get("handlers", {}):
        directory = log_dir or (settings.project_root / "logs")
        directory.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"]["filename"] = str(directory / "cews.log")
    else:
        config.get("handlers", {}).pop("file", None)
        for logger_config in [*config.get("loggers", {}).values(), config.get("root", {})]:
            handlers = logger_config.get("handlers")
            if isinstance(handlers, list) and "file" in handlers:
                handlers.remove("file")

    logging.config.dictConfig(config)
    install_redaction(settings.secret_values())
    LOGGER.debug("logging configured at level %s", level)


def install_redaction(secrets: Iterable[str]) -> None:
    """Attach (or update) a :class:`RedactingFilter` on all root and ``cews`` handlers."""
    secret_list = [s for s in secrets if s]
    loggers = [logging.getLogger(), logging.getLogger("cews")]
    for logger in loggers:
        for handler in logger.handlers:
            existing = [f for f in handler.filters if isinstance(f, RedactingFilter)]
            if existing:
                existing[0].update_secrets(secret_list)
            else:
                handler.addFilter(RedactingFilter(secret_list))
