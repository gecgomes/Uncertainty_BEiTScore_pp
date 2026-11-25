"""Filesystem locations used by the package and the scripts.

Every path is read from an environment variable so nothing is tied to one
machine. See configs/paths.example.sh.
"""
import os

DATA_ROOT = os.environ.get("BEITSCORE_DATA", "data")
MODELS_ROOT = os.environ.get("BEITSCORE_MODELS", "models")
OUTPUT_ROOT = os.environ.get("BEITSCORE_OUTPUT", "outputs")
FINETUNE_ROOT = os.path.join(OUTPUT_ROOT, "finetuning")
