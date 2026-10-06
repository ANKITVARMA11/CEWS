"""Unit tests for the typed CSV writer."""

from __future__ import annotations

import csv
import math
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from cews.exports.csv_export import Column, ExportError, format_cell, safe_text, write_csv

pytestmark = pytest.mark.unit


def cell(value: Any, kind: str) -> str:
    return format_cell(value, Column("x", kind))


def read(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


# --------------------------------------------------------------------------------------
# Cell formatting, one type at a time
# --------------------------------------------------------------------------------------
def test_missing_values_are_empty_in_every_type() -> None:
    for kind in ("int", "float", "text", "date", "datetime", "flag"):
        assert cell(None, kind) == ""


@pytest.mark.parametrize(
    ("value", "expected"), [(5, "5"), (5.0, "5"), ("7", "7"), (0, "0"), (-3, "-3")]
)
def test_whole_numbers(value: Any, expected: str) -> None:
    assert cell(value, "int") == expected


@pytest.mark.parametrize("value", [5.5, "abc", True, float("nan")])
def test_an_int_column_refuses_what_is_not_a_whole_number(value: Any) -> None:
    with pytest.raises(ExportError, match="cannot hold"):
        cell(value, "int")


def test_floats_use_a_full_stop_and_are_rounded_to_six_places() -> None:
    assert cell(1.23456789, "float") == "1.234568"
    assert cell(3, "float") == "3.0"
    assert "," not in cell(1234.5, "float")


@pytest.mark.parametrize("value", [float("nan"), math.inf, -math.inf])
def test_nan_and_infinity_become_blank_rather_than_the_words(value: float) -> None:
    assert cell(value, "float") == ""


def test_a_float_column_refuses_text() -> None:
    with pytest.raises(ExportError):
        cell("high", "float")


@pytest.mark.parametrize(("value", "expected"), [(True, "1"), (False, "0"), (1, "1"), (0, "0")])
def test_flags_are_one_or_zero(value: Any, expected: str) -> None:
    assert cell(value, "flag") == expected


def test_dates_are_iso() -> None:
    assert cell(date(2026, 8, 1), "date") == "2026-08-01"
    assert cell("2026-08-01", "date") == "2026-08-01"
    assert cell(datetime(2026, 8, 1, 13, 0, tzinfo=UTC), "date") == "2026-08-01"


def test_a_bad_date_is_refused() -> None:
    with pytest.raises(ExportError):
        cell("next tuesday", "date")


def test_timestamps_are_utc_iso() -> None:
    plus_five = timezone(timedelta(hours=5, minutes=30))
    assert cell(datetime(2026, 8, 1, 12, 0, tzinfo=plus_five), "datetime") == "2026-08-01T06:30:00Z"
    assert (
        cell(datetime(2026, 8, 1, 12, 0), "datetime") == "2026-08-01T12:00:00Z"
    )  # naive means UTC


def test_an_unknown_column_type_is_refused_when_declared() -> None:
    with pytest.raises(ExportError, match="unknown type"):
        Column("x", "decimal")


# --------------------------------------------------------------------------------------
# Text safety
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("start", ["=", "+", "-", "@"])
def test_text_that_could_run_as_a_formula_is_defused(start: str) -> None:
    assert cell(f'{start}HYPERLINK("http://evil")', "text").startswith("'")


def test_ordinary_text_is_not_altered() -> None:
    assert cell("CRISPR gene editing", "text") == "CRISPR gene editing"
    assert cell("mRNA - a platform", "text") == "mRNA - a platform"  # a dash inside is fine


def test_negative_numbers_are_not_mistaken_for_formulas() -> None:
    assert cell(-3.5, "float") == "-3.5" and cell(-2, "int") == "-2"


def test_line_breaks_and_control_characters_are_removed() -> None:
    assert safe_text("a\nb\r\nc\x00d\x1b") == "a b cd"


def test_unicode_survives() -> None:
    assert cell("Ångström — β-lactam", "text") == "Ångström — β-lactam"


# --------------------------------------------------------------------------------------
# Whole files
# --------------------------------------------------------------------------------------
COLUMNS = (Column("id", "int"), Column("name", "text"), Column("when", "date"))


def test_a_file_has_a_header_and_typed_rows(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    count = write_csv(path, COLUMNS, [{"id": 1, "name": "A, B", "when": date(2026, 1, 2)}])
    assert count == 1
    assert read(path) == [["id", "name", "when"], ["1", "A, B", "2026-01-02"]]


def test_an_empty_table_is_a_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    assert write_csv(path, COLUMNS, []) == 0
    assert read(path) == [["id", "name", "when"]]


def test_the_file_starts_with_a_byte_order_mark_for_excel(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    write_csv(path, COLUMNS, [])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_quotes_and_commas_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    write_csv(path, COLUMNS, [{"id": 1, "name": 'say "hi", ok', "when": None}])
    assert read(path)[1] == ["1", 'say "hi", ok', ""]


def test_a_row_missing_a_column_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="missing column"):
        write_csv(tmp_path / "t.csv", COLUMNS, [{"id": 1, "name": "x"}])


def test_extra_keys_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    write_csv(path, COLUMNS, [{"id": 1, "name": "x", "when": None, "extra": "ignored"}])
    assert read(path)[1] == ["1", "x", ""]


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    rows: list[dict[str, Any]] = [
        {"id": 1, "name": "ok", "when": None},
        {"id": "bad", "name": "x", "when": None},
    ]
    with pytest.raises(ExportError):
        write_csv(path, COLUMNS, rows)
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_an_existing_file_exactly_as_it_was(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    write_csv(path, COLUMNS, [{"id": 1, "name": "original", "when": None}])
    before = path.read_bytes()
    with pytest.raises(ExportError):
        write_csv(path, COLUMNS, [{"id": "bad", "name": "x", "when": None}])
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["t.csv"]


def test_rows_may_be_a_generator(tmp_path: Path) -> None:
    rows = ({"id": i, "name": str(i), "when": None} for i in range(3))
    assert write_csv(tmp_path / "t.csv", COLUMNS, rows) == 3


def test_the_output_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "er" / "t.csv"
    write_csv(path, COLUMNS, [])
    assert path.is_file()
