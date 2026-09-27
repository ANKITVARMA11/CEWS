"""Unit tests for cews.logging_config (configuration and secret redaction)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from cews.logging_config import (
    REDACTED,
    RedactingFilter,
    configure_logging,
    redact_secrets,
)
from cews.settings import Settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "secrets", "forbidden"),
    [
        ("calling with key abc123XYZ now", ["abc123XYZ"], "abc123XYZ"),
        ("GET /x?api_key=abc123&page=2", [], "abc123"),
        ("apikey: 'hunter2'", [], "hunter2"),
        ("password=p@ss, user=bob", [], "p@ss"),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", [], "eyJhbGciOi"),
        ("connect postgresql://cews:s3cret@localhost/db", [], "s3cret"),
        ("token=abc and secret=def", [], "abc"),
    ],
)
def test_redact_secrets_masks_credentials(text: str, secrets: list[str], forbidden: str) -> None:
    cleaned = redact_secrets(text, secrets)
    assert forbidden not in cleaned
    assert REDACTED in cleaned


def test_redact_secrets_leaves_ordinary_text_alone() -> None:
    text = "Fetched 120 records from pubmed in 3.2 seconds"
    assert redact_secrets(text, ["not-present"]) == text


def test_redact_secrets_ignores_empty_secret_values() -> None:
    assert redact_secrets("hello world", ["", ""]) == "hello world"


def test_longer_secret_is_masked_before_its_prefix() -> None:
    cleaned = redact_secrets("value=abcdef-long", ["abc", "abcdef-long"])
    assert "long" not in cleaned


def test_redacting_filter_rewrites_formatted_message() -> None:
    record = logging.LogRecord(
        "cews.test", logging.INFO, __file__, 1, "key is %s", ("topsecret",), None
    )
    assert RedactingFilter(["topsecret"]).filter(record) is True
    assert record.getMessage() == f"key is {REDACTED}"


def test_redacting_filter_keeps_clean_records_untouched() -> None:
    record = logging.LogRecord("cews.test", logging.INFO, __file__, 1, "plain %d", (5,), None)
    RedactingFilter(["topsecret"]).filter(record)
    assert record.getMessage() == "plain 5"
    assert record.args == (5,)


def test_filter_secrets_can_be_updated() -> None:
    log_filter = RedactingFilter([])
    log_filter.update_secrets(["later"])
    record = logging.LogRecord("cews.test", logging.INFO, __file__, 1, "x later y", None, None)
    log_filter.filter(record)
    assert "later" not in record.getMessage()


def test_configure_logging_writes_redacted_file(tmp_path: Path, settings: Settings) -> None:
    secured = settings.model_copy(
        update={"ncbi_api_key": SecretStr("file-leak-secret"), "log_level": "DEBUG"}
    )
    configure_logging(secured, log_dir=tmp_path / "logs")
    logger = logging.getLogger("cews.demo")
    logger.info("using key file-leak-secret for the request")
    logger.warning("second line api_key=%s", "another")
    for handler in logging.getLogger("cews").handlers:
        handler.flush()
    content = (tmp_path / "logs" / "cews.log").read_text(encoding="utf-8")
    assert "using key" in content
    assert "file-leak-secret" not in content
    assert "another" not in content


def test_configure_logging_applies_level(tmp_path: Path, settings: Settings) -> None:
    configure_logging(settings.model_copy(update={"log_level": "ERROR"}), log_dir=tmp_path)
    assert logging.getLogger("cews").level == logging.ERROR


def test_configure_logging_is_idempotent(tmp_path: Path, settings: Settings) -> None:
    configure_logging(settings, log_dir=tmp_path)
    configure_logging(settings, log_dir=tmp_path)
    for handler in logging.getLogger("cews").handlers:
        assert sum(isinstance(f, RedactingFilter) for f in handler.filters) == 1


def test_configure_logging_without_file_handler(settings: Settings) -> None:
    configure_logging(settings, enable_file=False)
    kinds = {type(h).__name__ for h in logging.getLogger("cews").handlers}
    assert "RotatingFileHandler" not in kinds
    assert "StreamHandler" in kinds


def test_configure_logging_rejects_invalid_config(tmp_path: Path, settings: Settings) -> None:
    bad = tmp_path / "logging.yaml"
    bad.write_text("just: a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dictConfig"):
        configure_logging(settings, config_path=bad, enable_file=False)
    broken = tmp_path / "broken.yaml"
    broken.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read"):
        configure_logging(settings, config_path=broken, enable_file=False)
