"""Delete all synthetic demo data (live data is never touched).

Equivalent to ``cews reset-demo``; extra command-line arguments are passed through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cews.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["reset-demo", *sys.argv[1:]]))
