"""Allow ``python -m layout_detect``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
