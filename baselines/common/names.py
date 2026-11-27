"""Metric names: how each appears in the table, its group there, and the runner that scores it."""
# name: (name in the table, group in the table, runner)
MODELS = {
    "beitscore-b": ("BEiTScore-B", "ours", "encoders"),
    "beitscore-l": ("BEiTScore-L", "ours", "encoders"),
    "beitscorepp-b": ("BEiTScore++-B", "ours", "encoders"),
    "beitscorepp-l": ("BEiTScore++-L", "ours", "encoders"),
    "clipscore-b": ("CLIP-S/ViT-B", "encoder", "encoders"),
    "clipscore-l": ("CLIP-S/ViT-L", "encoder", "encoders"),
    "pacscore++-b": ("PAC-S++/ViT-B", "encoder", "encoders"),
    "pacscore++-l": ("PAC-S++/ViT-L", "encoder", "encoders"),
    "specs": ("SPECS", "encoder", "encoders"),
    "fleur": ("FLEUR", "llm", "fleur"),
    "expert": ("EXPERT", "llm", "expert"),
    "vqascore": ("VQAScore", "llm", "vqascore"),
    "vela-paper-without_refs": ("VELA w/o references", "reported", None),
    "vela-paper-with_refs": ("VELA w/ references", "reported", None),
}

# Values reported by the VELA paper (Matsuda et al., EMNLP 2025, arXiv:2509.25818, Table 7) on the
# same test sets (TestA = dci_val, TestB = dci_test): Kendall's tau_c x100 for each rated dimension.
# "with_refs" is VELA itself (mean of five training runs); "without_refs" its reference-free variant.
# Both were fine-tuned on the training split of LongCap-Arena. Shown with --vela_paper, never ranked.
REPORTED_VELA = {
    "without_refs": {"dci_val": {"desc": 54.3, "rel": 36.1, "flu": 56.4},
                     "dci_test": {"desc": 46.4, "rel": 44.4, "flu": 33.4}},
    "with_refs": {"dci_val": {"desc": 56.4, "rel": 40.0, "flu": 57.4},
                  "dci_test": {"desc": 54.0, "rel": 52.3, "flu": 39.0}},
}
