# Power BI for CEWS

Streamlit is CEWS's primary dashboard. Power BI support is deliberately limited to **data,
documentation and measures**: CEWS writes clean CSV tables, and this folder tells you how to
build a report on them. **CEWS does not generate a `.pbix` report file.**

| File | What it is |
| --- | --- |
| `refresh_instructions.md` | Step by step: export, load into Power BI Desktop, refresh. |
| `data_model.md` | Every table and column, and the relationships between them. |
| `dax_measures.md` | Ready-made measures, including how to avoid double counting. |
| `dashboard_specification.md` | A suggested page layout that mirrors the Streamlit dashboard. |
| `powerbi_theme.json` | A colour-blind-safe theme (import it in Power BI Desktop). |

The short version:

```
cews export-powerbi
```

writes 13 tables and `refresh_metadata.json` to `data/exports/powerbi/`.

**Untested in Power BI Desktop.** The export is tested end to end, but the load query, the
relationships and the DAX in this folder were written and reviewed against the export, not run
inside Power BI Desktop (this project was built where Desktop is not available). Expect to
correct small things the first time you open it, and tell me what needed changing.

**Synthetic data.** Until real data is collected, every table holds invented demo data. Every
fact row carries `is_synthetic`, and `dax_measures.md` has a banner measure to put on every page.
