#!/usr/bin/env python
"""Encoder-based metrics: BEiTScore and BEiTScore++ (B, L), CLIPScore (B, L), PAC-S++ (B, L) and SPECS.

Runs in the environment of this repository, after `bash baselines/encoders/setup.sh`. It also
prepares the benchmark items that every runner scores, from the benchmark files of
scripts/prepare_data/benchmarks.py and scripts/download_release.py.

  python baselines/encoders/run_encoders.py --benchmarks vela vicr valse
  python baselines/encoders/run_encoders.py --benchmarks all --models beitscore-b beitscore-l
  python baselines/encoders/run_encoders.py --benchmarks all --prepare_only
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import benchmarks as B  # noqa: E402
from common import runner  # noqa: E402


def _factory(module, cls):
    def build(args):
        sys.path.insert(0, HERE)
        return getattr(__import__(module), cls)(args)
    return build


MODELS = {
    "beitscore-b": _factory("beitscore_scorer", "BEiTScoreB"),
    "beitscore-l": _factory("beitscore_scorer", "BEiTScoreL"),
    "beitscorepp-b": _factory("beitscore_scorer", "BEiTScorePPB"),
    "beitscorepp-l": _factory("beitscore_scorer", "BEiTScorePPL"),
    "clipscore-b": _factory("clipscore_scorer", "CLIPScoreB"),
    "clipscore-l": _factory("clipscore_scorer", "CLIPScoreL"),
    "pacscore++-b": _factory("pacscore_scorer", "PACScorePPB"),
    "pacscore++-l": _factory("pacscore_scorer", "PACScorePPL"),
    "specs": _factory("specs_scorer", "SPECS"),
}


def main():
    p = runner.base_parser(__doc__, MODELS)
    p.add_argument("--prepare_only", action="store_true", help="Only prepare the benchmark items")
    p.add_argument("--prepare_again", action="store_true",
                   help="Prepare the benchmark items again (after downloading a benchmark again)")
    for size in ("base", "large"):
        p.add_argument(f"--beitscore_{size}_checkpoint", default=None, help="Default: the downloaded release")
        p.add_argument(f"--beitscorepp_{size}_checkpoint", default=None, help="Default: the downloaded release")
    p.add_argument("--pacscore_repo", default=os.path.join(HERE, "repos", "pacscore"))
    p.add_argument("--specs_repo", default=os.path.join(HERE, "repos", "SPECS"))
    p.add_argument("--specs_checkpoint", default=None, help="Default: <specs_repo>/checkpoints/spec.pt")
    args = p.parse_args()
    runner.check_args(p, args, MODELS)

    if args.prepare_again:
        for bench in args.benchmarks:
            path = os.path.join(args.results, "items", f"{bench}.json")
            if os.path.exists(path):
                os.remove(path)
    items = runner.load_items(args, prepare=lambda b: B.prepare(b, args.results, args.data))
    if args.prepare_only:
        return
    failed = runner.score_all(args, MODELS, items)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
