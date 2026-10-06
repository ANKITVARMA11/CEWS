"""Writing typed, predictable CSV files.

Everything that leaves CEWS as a spreadsheet-style file goes through here, so the rules are in
one place and cannot drift between exports:

* **Explicit types.** Each column is declared as int, float, text, date, datetime or flag, and a
  value that does not fit its type is an error, not a silently wrong cell.
* **One spelling for everything.** Dates are ``YYYY-MM-DD``, timestamps are ISO 8601 in UTC,
  numbers use a full stop, a flag is ``1`` or ``0``, and a missing value is an empty cell.
* **Headers always.** A table with no rows is a header-only file, so a report built on it does
  not break the first time a table happens to be empty.
* **Text cannot run as a formula.** A text cell that begins with ``=``, ``+``, ``-`` or ``@`` is
  prefixed with an apostrophe, so opening an export in Excel cannot execute anything written in
  a record title by an outside source.
* **Files appear whole.** Each file is written beside its destination and moved into place, so a
  program reading the folder never sees a half-written file.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

COLUMN_TYPES = ("int", "float", "text", "date", "datetime", "flag")
FLOAT_PLACES = 6
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
_CONTROL = {code: None for code in list(range(0, 9)) + [11, 12] + list(range(14, 32)) + [127]}


class ExportError(ValueError):
    """Raised when data cannot be written as declared."""


@dataclass(frozen=True)
class Column:
    """One column: its name and its type (one of :data:`COLUMN_TYPES`)."""

    name: str
    type: str

    def __post_init__(self) -> None:
        if self.type not in COLUMN_TYPES:
            raise ExportError(f"column {self.name!r} has unknown type {self.type!r}")


def safe_text(value: Any) -> str:
    """Text with control characters removed, line breaks turned into spaces, and no leading
    character that a spreadsheet would treat as the start of a formula."""
    text = str(value).translate(_CONTROL)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return "'" + text if text.startswith(FORMULA_STARTS) else text


def format_cell(value: Any, column: Column) -> str:
    """One value as it will appear in the file.

    Raises:
        ExportError: if the value cannot be represented as the column's declared type.
    """
    if value is None:
        return ""
    kind = column.type
    try:
        if kind == "text":
            return safe_text(value)
        if kind == "flag":
            return "1" if bool(value) else "0"
        if kind == "int":
            if isinstance(value, bool) or not float(value).is_integer():
                raise ValueError("not a whole number")
            return str(int(value))
        if kind == "float":
            number = float(value)
            return (
                ""
                if math.isnan(number) or math.isinf(number)
                else repr(round(number, FLOAT_PLACES))
            )
        if kind == "date":
            return (value.date() if isinstance(value, datetime) else _as_date(value)).isoformat()
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExportError(f"column {column.name!r} ({kind}) cannot hold {value!r}: {exc}") from exc


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def write_csv(path: Path, columns: Sequence[Column], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write ``rows`` to ``path`` (UTF-8, header first) and return how many were written.

    A row missing a declared column is an error; extra keys are ignored. On any failure the
    partly-written file is removed and an existing file at ``path`` is left exactly as it was.

    Raises:
        ExportError: for a missing column or a value that does not fit its type.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    try:
        # utf-8-sig writes a byte-order mark, which Excel needs to read UTF-8 correctly and
        # which Power BI and Python's csv module both accept.
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([column.name for column in columns])
            for row in rows:
                missing = [column.name for column in columns if column.name not in row]
                if missing:
                    raise ExportError(f"{path.name}: row is missing column(s) {missing}")
                writer.writerow([format_cell(row[column.name], column) for column in columns])
                count += 1
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count
