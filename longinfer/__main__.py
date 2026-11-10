# -*- coding: utf-8 -*-
"""Allow `python -m longinfer`."""
import sys

from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
