"""Run refresh cycles on an interval until stopped. Identical to ``cews run-scheduler``.

python scripts/run_scheduler.py            # first cycle after one interval
python scripts/run_scheduler.py --now      # first cycle immediately
"""

from __future__ import annotations

import sys

from cews.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-scheduler", *sys.argv[1:]]))
