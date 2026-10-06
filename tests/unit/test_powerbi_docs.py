"""The Power BI documentation must describe the export that actually exists.

A column renamed in the code but not in the docs would break a report silently, so these
tests read the documentation and check it against the table definitions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cews.exports.powerbi_export import TABLE_NAMES, TABLES

pytestmark = pytest.mark.unit

DOCS = Path(__file__).resolve().parents[2] / "dashboards" / "powerbi"
COLUMNS = {spec.name: {column.name for column in spec.columns} for spec in TABLES}


def section(text: str, table: str) -> str:
    match = re.search(rf"^### {table}\n(.*?)(?=^### |^## |\Z)", text, re.S | re.M)
    assert match, f"data_model.md has no section for {table}"
    return match.group(1)


def test_the_data_model_documents_every_table() -> None:
    text = (DOCS / "data_model.md").read_text(encoding="utf-8")
    for name in TABLE_NAMES:
        section(text, name)


@pytest.mark.parametrize("table", TABLE_NAMES)
def test_the_data_model_documents_every_column_of_a_table(table: str) -> None:
    body = section((DOCS / "data_model.md").read_text(encoding="utf-8"), table)
    documented = set(re.findall(r"^\| `(\w+)` \|", body, re.M))
    assert (
        documented == COLUMNS[table]
    ), f"{table}: undocumented {COLUMNS[table] - documented}, unknown {documented - COLUMNS[table]}"


def test_every_column_type_in_the_docs_matches_the_code() -> None:
    text = (DOCS / "data_model.md").read_text(encoding="utf-8")
    for spec in TABLES:
        body = section(text, spec.name)
        for column in spec.columns:
            row = re.search(rf"^\| `{column.name}` \| (\w+) \|", body, re.M)
            assert row and row.group(1) == column.type, (spec.name, column.name)


def test_every_dax_column_reference_exists() -> None:
    text = (DOCS / "dax_measures.md").read_text(encoding="utf-8")
    references = set(re.findall(r"\b((?:dim|fact)_\w+)\[(\w+)\]", text))
    assert references, "the DAX file should reference the model"
    for table, column in references:
        assert table in COLUMNS, f"DAX refers to unknown table {table}"
        assert column in COLUMNS[table], f"DAX refers to unknown column {table}[{column}]"


def test_every_dax_measure_references_only_measures_that_are_defined() -> None:
    text = (DOCS / "dax_measures.md").read_text(encoding="utf-8")
    defined = set(re.findall(r"^([A-Z][\w %-]*?) =\s*$|^([A-Z][\w %-]*?) = ", text, re.M))
    names = {a or b for a, b in defined}
    used = set(re.findall(r"(?<![\w\]])\[([A-Z][\w %-]+)\]", text))
    unknown = {name for name in used if name not in names}
    assert not unknown, f"measures used but never defined: {unknown}"


def test_the_dax_pins_the_activity_period_type() -> None:
    """The double-counting guard: every direct sum of activity must name a period type."""
    text = (DOCS / "dax_measures.md").read_text(encoding="utf-8")
    for block in re.findall(r"SUM \( fact_activity\[activity_count\] \)(.*?)\n\)", text, re.S):
        assert 'period_type] = "month"' in block


def test_the_metric_paths_the_dax_uses_are_ones_the_evaluation_produces() -> None:
    text = (DOCS / "dax_measures.md").read_text(encoding="utf-8")
    used = set(re.findall(r'metric_path\] = "([\w.]+)"', text))
    assert used == {"mean_precision_at_k", "mean_spearman", "precision"}


def test_the_docs_do_not_claim_to_generate_a_report_file() -> None:
    text = (DOCS / "README.md").read_text(encoding="utf-8")
    assert "does not generate a `.pbix`" in text
    assert "Untested in Power BI Desktop" in text


def test_the_load_query_names_the_metadata_file_and_the_culture() -> None:
    text = (DOCS / "refresh_instructions.md").read_text(encoding="utf-8")
    assert "refresh_metadata.json" in text and '"en-US"' in text and "Encoding = 65001" in text


def test_the_theme_is_valid_json_with_the_colour_blind_palette() -> None:
    import json

    theme = json.loads((DOCS / "powerbi_theme.json").read_text(encoding="utf-8"))
    assert theme["dataColors"][0] == "#0072B2"
