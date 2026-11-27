"""Code shared by the baseline runners and make_table.py.

The repository root is put on the import path so that every environment, including those of
FLEUR, EXPERT and VQAScore, which do not install the beitscore package, can import the
numpy-only module beitscore.vela (importing beitscore itself loads nothing heavy).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
