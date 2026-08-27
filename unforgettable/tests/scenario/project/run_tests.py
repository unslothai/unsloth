#!/usr/bin/env python
"""World-judge harness. Unittest, not pytest — so the parent suite cannot collect us."""

from __future__ import annotations

import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover("ledger_tests", pattern = "check_*.py")
    result = unittest.TextTestRunner(verbosity = 2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
