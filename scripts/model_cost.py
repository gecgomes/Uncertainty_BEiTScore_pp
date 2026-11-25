"""Parameters and GFLOPs of BEiTScore and BEiTScore++ (efficiency comparisons of both papers).

    python scripts/model_cost.py                       # base and large, both metrics
    python scripts/model_cost.py --tokens 64 --device cuda --timing

Counts the FLOPs of one forward pass for one image (480 x 480) and one caption of
--tokens tokens with torch.utils.flop_counter (multiply-adds counted as two FLOPs).
No checkpoint is needed: the cost does not depend on the weights. With --timing, also
reports the median latency over --repeats passes.
"""
import argparse
import time

import pandas as pd
import torch
from timm.models import create_model
from torch.utils.flop_counter import FlopCounterMode

import beitscore.models  # noqa: F401  (registers the model configs)


def cost(model_size, task, tokens, device, timing, repeats):
    kwargs = dict(chained_construction=True, quantiles=[0.9, 0.85, 0.8, 0.75, 0.7]) if task == "quant" else {}
    model = create_model(f"beit3_{model_size}_patch16_480_{task}", pretrained=False, drop_path_rate=0.0,
                         vocab_size=64010, checkpoint_activations=False, **kwargs).to(device).eval()
    image = torch.randn(1, 3, 480, 480, device=device)
    text = torch.randint(3, 64000, (1, tokens), device=device)
    mask = torch.zeros_like(text)
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(image=image, text_description=text, padding_mask=mask, only_inference=True)
    row = {"metric": "BEiTScore++" if task == "quant" else "BEiTScore", "size": model_size,
           "params_M": params / 1e6, "GFLOPs": counter.get_total_flops() / 1e9}
    if timing:
        times = []
        with torch.no_grad():
            for _ in range(repeats):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(image=image, text_description=text, padding_mask=mask, only_inference=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        row["latency_ms_median"] = float(pd.Series(times[1:] or times).median())
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["base", "large"])
    parser.add_argument("--tokens", type=int, default=20, help="Caption length in tokens, BOS and EOS included")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    rows = [cost(s, t, args.tokens, args.device, args.timing, args.repeats)
            for s in args.sizes for t in ("regression", "quant")]
    print(pd.DataFrame(rows).round(2).to_string(index=False))
