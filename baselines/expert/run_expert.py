#!/usr/bin/env python
"""EXPERT with LLaVA-1.5-13B, in its own environment (baselines/expert/setup.sh), on the LLaVA code the EXPERT repository
pins: LLaVA 1.2.2 (torch 2.1.2, transformers 4.37.2).

  baselines/expert/.venv/bin/python baselines/expert/run_expert.py --benchmarks vela
  baselines/expert/.venv/bin/python baselines/expert/run_expert.py --benchmarks flickr8k-expert composite   # check against the published numbers

The benchmark items must have been prepared by baselines/encoders/run_encoders.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import runner  # noqa: E402


def build(args):
    from common.llava_judge import EXPERT
    return EXPERT(args)


MODELS = {"expert": build}


def main():
    p = runner.base_parser(__doc__, MODELS)
    p.add_argument("--llava_dir", default=os.path.join(HERE, "EXPERT", "LLaVA"), help="The LLaVA submodule of EXPERT")
    args = p.parse_args()
    runner.check_args(p, args, MODELS)
    failed = runner.score_all(args, MODELS, runner.load_items(args))
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
