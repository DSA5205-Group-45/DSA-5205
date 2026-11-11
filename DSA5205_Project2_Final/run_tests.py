#!/usr/bin/env python
"""Convenience wrapper around pytest.

Examples
--------
python run_tests.py                 # fast suite (notebook execution skipped)
python run_tests.py --notebook      # include the full notebook execution test
python run_tests.py -k leakage      # pass any pytest selector
"""
from __future__ import annotations

import os
import subprocess
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    if "--notebook" in argv:
        argv.remove("--notebook")
        env["RUN_NOTEBOOK_TEST"] = "1"
    command = [sys.executable, "-m", "pytest", "-q", *argv]
    print(" ".join(command))
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
