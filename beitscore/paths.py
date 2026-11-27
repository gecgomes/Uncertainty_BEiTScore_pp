"""Filesystem locations used by the package and the scripts.

Every path is read from an environment variable, set by `source configs/paths.sh`, so nothing is
tied to one machine. Without them, the folders data/, models/ and outputs/ at the root of the
repository are used, as configs/paths.sh does by default (or of the current folder, when the
package is installed outside the repository).
"""
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE = _REPO if os.path.exists(os.path.join(_REPO, "pyproject.toml")) else os.getcwd()

DATA_ROOT = os.environ.get("BEITSCORE_DATA", os.path.join(_BASE, "data"))
MODELS_ROOT = os.environ.get("BEITSCORE_MODELS", os.path.join(_BASE, "models"))
OUTPUT_ROOT = os.environ.get("BEITSCORE_OUTPUT", os.path.join(_BASE, "outputs"))
FINETUNE_ROOT = os.path.join(OUTPUT_ROOT, "finetuning")
