"""Make the flat `xrec_ref/` layout importable from the test suite and from `python -m`."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, "xrec_ref")
if REF not in sys.path:
    sys.path.insert(0, REF)
