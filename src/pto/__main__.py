"""``python -m pto`` entry point (same as the ``pto`` console script)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
