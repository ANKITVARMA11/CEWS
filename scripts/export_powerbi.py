"""Write the Power BI CSV export. Identical to ``cews export-powerbi``.

python scripts/export_powerbi.py
python scripts/export_powerbi.py --out C:\\reports\\cews --env-file .env
"""

from __future__ import annotations

import sys

from cews.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export-powerbi", *sys.argv[1:]]))
