"""Collect every enabled source once (manual refresh).

Equivalent to ``cews fetch``; options such as ``--dry-run``, ``--source ID`` and ``--full``
are passed through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cews.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["fetch", *sys.argv[1:]]))
