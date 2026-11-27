"""LaTeX table: one row per model, one column per benchmark (several for VELA), best value in
bold and second best underlined per column. Models are grouped as in the BEiTScore tables:
LLM-based, encoder-based, then BEiTScore."""
from .names import MODELS
from beitscore.vela import GROUP_NAMES, TARGET_NAMES

BENCH_NAMES = {"vela": "LongCap-Arena", "vicr": "VICR", "polaris": "Polaris", "nebula": "Nebula",
               "flickr8k-expert": "Flickr8K-Expert", "flickr8k-cf": "Flickr8K-CF", "composite": "Composite",
               "genai": "GenAI-Bench", "nocaps-foil": "nocaps-FOIL", "valse": "VALSE", "sugarcrepe": "SugarCrepe",
               "longcapvlcp": "LongCapVLCP", "winoground": "Winoground"}
GROUPS = [("reported", "Reported by \\citet{matsuda2025vela}"), ("llm", "LLM-based Methods"),
          ("encoder", "Encoder-based Methods"), ("ours", None)]


def _fmt(v, rank):
    if v is None:
        return "--"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if rank == 0 else f"\\underline{{{s}}}" if rank == 1 else s


def latex(cells, models, columns, metric_names, caption_note=""):
    """cells: {(model, column): value}; columns: [(benchmark, sub)] in order, sub "" or a VELA target."""
    ranks = {}
    for col in columns:
        ranked = [m for m in models if MODELS[m][1] != "reported"]  # values quoted from a paper are not ranked
        vals = sorted({round(cells[(m, col)], 1) for m in ranked if cells.get((m, col)) is not None}, reverse=True)
        for m in ranked:
            v = cells.get((m, col))
            ranks[(m, col)] = vals.index(round(v, 1)) if v is not None and round(v, 1) in vals[:2] else None

    benches = list(dict.fromkeys(b for b, _ in columns))
    three = any("|" in s for _, s in columns)
    head1, head2, head3, rules1, rules2, start = ["\\textbf{Model}"], [""], [""], [], [], 2
    for b in benches:
        subs = [s for bb, s in columns if bb == b]
        label = f"\\textbf{{{BENCH_NAMES.get(b, b)}}}"
        if subs == [""]:
            head1.append(label)
            head2.append(metric_names[b])
            head3.append("")
        else:
            head1.append(f"\\multicolumn{{{len(subs)}}}{{c}}{{{label}}}")
            rules1.append(f"\\cmidrule(lr){{{start}-{start + len(subs) - 1}}}")
            targets = list(dict.fromkeys(s.split("|")[0] for s in subs))
            pos = start
            for t in targets:
                tsubs = [s for s in subs if s.split("|")[0] == t]
                if len(tsubs) > 1:
                    head2.append(f"\\multicolumn{{{len(tsubs)}}}{{c}}{{{TARGET_NAMES[t]}}}")
                    rules2.append(f"\\cmidrule(lr){{{pos}-{pos + len(tsubs) - 1}}}")
                else:
                    head2.append(TARGET_NAMES[t])
                head3 += [GROUP_NAMES.get(s.split("|")[1], s.split("|")[1]) if "|" in s else "" for s in tsubs]
                pos += len(tsubs)
        start += len(subs)
    ncol = 1 + len(columns)
    lines = ["\\begin{table*}[t]", "\\centering", "\\small", "\\resizebox{\\textwidth}{!}{%",
             f"\\begin{{tabular}}{{l{'c' * len(columns)}}}", "\\toprule",
             " & ".join(head1) + " \\\\"]
    if rules1:
        lines.append(" ".join(rules1))
    lines.append(" & ".join(head2) + " \\\\")
    if three:
        if rules2:
            lines.append(" ".join(rules2))
        lines.append(" & ".join(head3) + " \\\\")
    lines.append("\\midrule")
    for group, title in GROUPS:
        members = [m for m in models if MODELS[m][1] == group]
        if not members:
            continue
        if title:
            lines += ["\\rowcolor{gray!4}", f"\\multicolumn{{{ncol}}}{{c}}{{{title}}} \\\\", "\\midrule"]
        for m in members:
            lines.append(" & ".join([MODELS[m][0]] + [_fmt(cells.get((m, c)), ranks.get((m, c))) for c in columns]) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    short = "Comparison of reference-free metrics across the selected benchmarks"
    full = ("Comparison of reference-free metrics across the selected benchmarks. "
            f"{caption_note}The highest score is highlighted using \\textbf{{bold}}, and the second best is "
            "highlighted using an \\underline{underline}.")
    lines += ["\\end{tabular}}", f"\\caption[{short}]{{{full}}}", "\\label{tab:baselines}", "\\end{table*}"]
    return "\n".join(lines) + "\n"
