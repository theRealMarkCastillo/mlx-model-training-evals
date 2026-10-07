"""Entry point: `uv run python main.py <command>`. See `main.py --help`."""

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
