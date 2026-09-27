"""Load deterministic SYNTHETIC demo data into the database.

Equivalent to ``cews seed-demo``; extra command-line arguments are passed through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cews.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["seed-demo", *sys.argv[1:]]))
