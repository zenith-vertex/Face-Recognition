"""Main entry point for the AI Face Recognition System CLI."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the CLI application."""
    from .presentation.cli import main as cli_main

    try:
        cli_main()
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        print(f"[red]Fatal error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
