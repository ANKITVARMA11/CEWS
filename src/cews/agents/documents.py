"""Reading an uploaded document into plain text — the whole thing, no chunking.

CEWS's other AI features (topic discovery, announcement extraction) work by splitting text into
pieces and matching or embedding each piece. This module deliberately does not do that. When a
person attaches their own document to a chat, the point is for the model to read it directly, the
way a person would hand a colleague a printout: the whole document goes into the conversation in
one message, and the model reasons over all of it at once with no retrieval step in between.

The one limit is size: a local model's context window is finite, so the text is capped at
``AGENT_DOCUMENT_CHAR_LIMIT`` characters. Going over that limit is never silent — the returned
:class:`DocumentText` always says how much of the document is actually included, so an agent
(and the person reading its answer) knows to treat anything beyond that point as unseen, rather
than assuming the model read the whole thing when it did not.

Supported formats are the common office ones a business document actually arrives in: PDF, Word
(.docx), Excel (.xlsx), CSV and plain text/Markdown. Nothing here uses machine-learning layout
models (compare `docling <https://github.com/docling-project/docling>`_, whose default install
pulls in torch and multi-GB models) — these are pure-Python text extractors, chosen to match this
project's offline-first, low-resource design.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from docx import Document as WordDocument
from pypdf import PdfReader

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".txt", ".md"})
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB; a sanity limit on what gets read into memory at all


class DocumentReadError(ValueError):
    """Raised when a document cannot be read as asked."""


@dataclass(frozen=True)
class DocumentText:
    """The text read from one document, and whether any of it had to be left out."""

    filename: str
    text: str
    full_length: int  # the length of the extracted text BEFORE any truncation
    truncated: bool

    @property
    def note(self) -> str:
        """A one-line, always-true statement of how much of the document is included.

        An agent puts this next to the document itself in the conversation, so the model (and,
        through its answer, the person) never mistakes a truncated document for a complete one.
        """
        if not self.truncated:
            return f"'{self.filename}': full document included ({self.full_length:,} characters)."
        return (
            f"'{self.filename}': only the first {len(self.text):,} of {self.full_length:,} "
            "characters are included below; anything after that point was not read."
        )


def _read_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises several distinct exception types for a bad PDF
        raise DocumentReadError(f"could not read this PDF: {exc}") from exc
    return "\n\n".join(pages).strip()


def _read_docx(data: bytes) -> str:
    try:
        document = WordDocument(io.BytesIO(data))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
    except Exception as exc:
        raise DocumentReadError(f"could not read this Word document: {exc}") from exc
    return "\n".join(part for part in parts if part.strip()).strip()


def _read_xlsx(data: bytes) -> str:
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        sheets: list[str] = []
        for sheet in workbook.worksheets:
            rows = [
                " | ".join("" if cell is None else str(cell) for cell in row)
                for row in sheet.iter_rows(values_only=True)
            ]
            sheets.append(f"# Sheet: {sheet.title}\n" + "\n".join(rows))
    except Exception as exc:
        raise DocumentReadError(f"could not read this spreadsheet: {exc}") from exc
    return "\n\n".join(sheets).strip()


def _read_csv(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DocumentReadError(f"could not read this CSV: {exc}") from exc
    return "\n".join(" | ".join(row) for row in rows).strip()


def _read_plain_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise DocumentReadError(f"this file is not readable UTF-8 text: {exc}") from exc


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".xlsm": _read_xlsx,
    ".csv": _read_csv,
    ".txt": _read_plain_text,
    ".md": _read_plain_text,
}


def read_document(filename: str, data: bytes, *, char_limit: int) -> DocumentText:
    """Extract the full text of one uploaded file.

    Args:
        filename: the original filename, used only to pick a reader by its suffix and to label
            the result; the file is never written to disk here (see
            :mod:`cews.agents.session` for how an upload's temporary file is handled).
        data: the file's raw bytes.
        char_limit: the most characters of extracted text to keep (see the module docstring).

    Raises:
        DocumentReadError: for an unsupported file type, a file over ``MAX_UPLOAD_BYTES``, an
            empty result, or a file that cannot be parsed as its declared type.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentReadError(
            f"'{filename}' is {len(data) / 1_000_000:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES / 1_000_000:.0f} MB limit for a single upload"
        )
    suffix = Path(filename).suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise DocumentReadError(
            f"'{filename}' has an unsupported file type ({suffix or 'no extension'}); "
            f"supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    text = reader(data)
    if not text:
        raise DocumentReadError(
            f"'{filename}' produced no readable text (it may be scanned or empty)"
        )
    truncated = len(text) > char_limit
    return DocumentText(
        filename=filename,
        text=text[:char_limit] if truncated else text,
        full_length=len(text),
        truncated=truncated,
    )
