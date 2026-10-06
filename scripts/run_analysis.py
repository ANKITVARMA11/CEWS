"""Re-run the whole analysis on the data already collected (no fetching).

Identical to ``cews refresh --skip-fetch``: normalize, competitors, features, scores, forecasts,
insights and the Power BI export, in order.
"""

from __future__ import annotations

import sys

from cews.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["refresh", "--skip-fetch", *sys.argv[1:]]))
