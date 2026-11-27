#!/usr/bin/env python
"""FLEUR with LLaVA-1.5-13B, in its own environment (baselines/fleur/setup.sh), on the LLaVA code the FLEUR repository
pins: LLaVA 1.1.3 (torch 2.0.1, transformers 4.31.0).

  baselines/fleur/.venv/bin/python baselines/fleur/run_fleur.py --benchmarks vela
  baselines/fleur/.venv/bin/python baselines/fleur/run_fleur.py --benchmarks flickr8k-expert composite   # check against the published numbers

The benchmark items must have been prepared by baselines/encoders/run_encoders.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import runner  # noqa: E402


def build(args):
    from common.llava_judge import FLEUR
    return FLEUR(args)


MODELS = {"fleur": build}


def main():
    p = runner.base_parser(__doc__, MODELS)
    p.add_argument("--llava_dir", default=os.path.join(HERE, "FLEUR", "LLaVA"), help="The LLaVA submodule of FLEUR")
    args = p.parse_args()
    runner.check_args(p, args, MODELS)
    failed = runner.score_all(args, MODELS, runner.load_items(args))
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
