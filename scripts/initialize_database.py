"""Create or upgrade the CEWS database schema and load the topic taxonomy.

Equivalent to ``cews db-init``; extra command-line arguments are passed through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cews.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["db-init", *sys.argv[1:]]))
