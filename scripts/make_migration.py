#!/usr/bin/env python3
"""Generate a new Alembic migration."""
import subprocess
import sys

if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "auto migration"
    subprocess.run(["alembic", "revision", "--autogenerate", "-m", message], check=True)
