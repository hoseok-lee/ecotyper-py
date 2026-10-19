"""Write the ``--help`` text of every ``ecotyper-py`` subcommand to docs/_generated/cli_<cmd>.txt."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

from ecotyper import cli

OUT = Path(__file__).resolve().parents[1] / "_generated"
COMMANDS = ["recover-bulk", "recover-scrna", "recover-visium", "discover", "discover-scrna"]


def _help(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        cli.main(args)
    return buf.getvalue()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cli_main.txt").write_text(_help(["--help"]))
    for c in COMMANDS:
        (OUT / f"cli_{c}.txt").write_text(_help([c, "--help"]))


if __name__ == "__main__":
    build()
