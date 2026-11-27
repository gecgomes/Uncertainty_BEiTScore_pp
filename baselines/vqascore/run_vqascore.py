#!/usr/bin/env python
"""VQAScore with CLIP-FlanT5-XXL, through the t2v_metrics package, in its own environment
(baselines/vqascore/setup.sh).

  baselines/vqascore/.venv/bin/python baselines/vqascore/run_vqascore.py --benchmarks vela

The benchmark items must have been prepared by baselines/encoders/run_encoders.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import runner  # noqa: E402


def build(args):
    from scorer import VQAScore
    return VQAScore(args)


MODELS = {"vqascore": build}


def main():
    p = runner.base_parser(__doc__, MODELS)
    args = p.parse_args()
    runner.check_args(p, args, MODELS)
    failed = runner.score_all(args, MODELS, runner.load_items(args))
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
