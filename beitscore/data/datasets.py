"""Benchmark loading and the BEiT-3 image-text dataset.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from .collate import merge_batch_tensors_by_dict_key
from .sampler import ClusterBatchSampler

# Nominal rating scale of each human-judgement benchmark (BEiTScore++ Table 3).
# Test labels are mapped to [0, 1] with these ranges, so the mapping does not
# depend on which ratings happen to occur in the test split. Benchmarks that are
# not listed fall back to the observed minimum and maximum, with a warning.
NOMINAL_SCALES = {
    "vicr": (1.0, 5.0),
    "polaris": (0.0, 1.0),
    "nebula": (0.0, 1.0),
    "flickr8k-expert": (1.0, 4.0),
    "composite": (1.0, 5.0),
}

# Evaluation benchmarks: each lives in $BEITSCORE_DATA/<name>/<name>.pkl with its images in
# $BEITSCORE_DATA/<name>/images/ (written by scripts/prepare_data/benchmarks.py, or by
# scripts/download_release.py for LongCapVLCP). name -> (task, Winoground-style tags).
BENCHMARKS = {
    "vicr": ("correlation", False),
    "polaris": ("correlation", False),
    "nebula": ("correlation", False),
    "flickr8k-expert": ("correlation", False),
    "flickr8k-cf": ("correlation", False),
    "composite": ("correlation", False),
    "genai": ("correlation", False),
    "nocaps-foil": ("foil", False),
    "valse": ("foil", False),
    "sugarcrepe": ("foil", False),
    "winoground": ("foil", True),
    "longcapvlcp": ("foil", False),
}
CORRELATION_BENCHMARKS = [n for n, (task, _) in BENCHMARKS.items() if task == "correlation"]
ERROR_BENCHMARKS = [n for n, (task, _) in BENCHMARKS.items() if task == "foil"]


def canonical_benchmark(name):
    """The standard name of a benchmark: lower case, as in BENCHMARKS."""
    return str(name).strip().lower()


def benchmark_file(name, base_path):
    name = canonical_benchmark(name)
    return os.path.join(base_path, name, f"{name}.pkl")


def transform_bool(_str):
    if isinstance(_str,str):
        if _str.lower() == "true":
            return True
        elif _str.lower() == "false":
            return False
    return _str


def resolve_image_path(path, data_path, prefix=""):
    """Image paths written by scripts/prepare_data/ are relative to $BEITSCORE_DATA (plus
    `prefix`); older files hold absolute paths, of which the part after the last "data"
    folder is kept."""
    path = str(path)
    if not os.path.isabs(path):
        return os.path.join(data_path, prefix, path)
    if os.path.exists(path):  # absolute paths written by scripts/get_images.py
        return path
    return data_path + ("/" + prefix if prefix else "") + "/" + path.split("data")[-1].lstrip("/")


def _dataset_preprocess(dataset, data_path, flickr_flag):
    if flickr_flag:
        dataset = dataset.groupby(["image_ids", "caption"]).agg(
            human_scores=('human_scores', list),
            image_path=('image_path', 'first'),
            data_type=('data_type', 'first'),
            refs=('refs', 'first')
        ).reset_index()
        dataset["image_path"] = dataset["image_path"].apply(lambda x: resolve_image_path(x.lstrip("/"), data_path, "flickr8k"))
    else:
        dataset["image_path"] = dataset["image_path"].apply(lambda x: resolve_image_path(x, data_path))

    if "human_scores" not in dataset.columns:
        if "all_human_scores" in dataset.columns:
            dataset["human_scores"] = dataset["all_human_scores"].to_list()
        elif "score" in dataset.columns:
            dataset["human_scores"] = dataset["score"].to_list()
        elif "scores" in dataset.columns:
            dataset["human_scores"] = dataset["scores"].to_list()
    if "caption" in dataset.columns:
        dataset["candidates"] = dataset["caption"].to_list()

    return dataset



def get_data(benchmark_name, base_path, split, seed=0):

    benchmark_name = canonical_benchmark(benchmark_name)
    if benchmark_name in BENCHMARKS:
        benchmark_path = benchmark_file(benchmark_name, base_path)
        task, wino_tag = BENCHMARKS[benchmark_name]
    elif benchmark_name == "stage1":
        benchmark_path = f"{base_path}/Stage1/stage1.pkl"  # written by scripts/download_release.py stage1
        wino_tag = False
        task = "stage1"
    elif benchmark_name == "regression":
        benchmark_path = f"{base_path}/stage2/stage2.pkl"  # written by scripts/download_release.py stage2
        wino_tag = False
        task = "regression"
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")

    print(f"Loading {split} dataset: {benchmark_path}")
    if not os.path.exists(benchmark_path):
        hint = ("python scripts/prepare_data/benchmarks.py download " + benchmark_name if benchmark_name in BENCHMARKS
                else "python scripts/download_release.py stage2" if task == "regression"
                else "python scripts/download_release.py stage1")
        raise FileNotFoundError(f"{benchmark_path} not found; prepare it with: {hint}")
    df = joblib.load(benchmark_path)
    if task in ["regression", "stage1"]:
        df = df.sample(frac=1, random_state=seed)

    if wino_tag:
        df["tag_combination"] = df["pair_id"].apply(lambda x: x[:2])
        df["pair_id"] = df["pair_id"].apply(lambda x: x[2:])

    if "subset" in df.columns:
        if ("test" not in list(df["subset"].unique())) and split == "test":
            split = "val"
        df = df[df["subset"] == split].reset_index(drop=True)

    if "split" in df.columns:
        if ("test" not in list(df["split"].unique())) and split == "test":
            split = "val"
        df = df[df["split"] == split].reset_index(drop=True)

    if task == "foil":
        foil_list = df["foil"].apply(transform_bool).tolist()
        
    if "caption" not in df.columns:
        if "candidates" in df.columns:
            df["caption"] = df["candidates"]
        elif "cand" in df.columns:
            df["caption"] = df["cand"]
        else:
            raise ValueError(f"No captions in dataset: {benchmark_name}")
    
    if "data_type" not in df.columns:
        if "task" in df.columns:
            df["data_type"] = df["task"]
        elif "domain" in df.columns:
            df["data_type"] = df["domain"]
        else:
            df["data_type"] = benchmark_name
    
    if "task" not in df.columns:
        df["task"] = task
    
    if "pair_id" not in df.columns:
        if "pair_ids" in df.columns:
            df["pair_id"] = df["pair_ids"]
        else:
            df["pair_id"] = range(len(df))
    
    if "image_path" not in df.columns:
        df["image_path"] = df["image_ids"].apply(lambda x: "/".join(benchmark_path.split("/")[:-1]) + "/images/" + x)
    
    if task in ("foil", "stage1"):
        # files written by scripts/prepare_data/ store paths relative to $BEITSCORE_DATA
        relative = ~df["image_path"].astype(str).map(os.path.isabs)
        df.loc[relative, "image_path"] = [os.path.join(base_path, p) for p in df.loc[relative, "image_path"]]

    if task == "foil":
        if (not wino_tag) and len(foil_list) and isinstance(foil_list[0], (bool, np.bool_)):
            df_true = df[df["foil"] == False].sort_values("pair_id").reset_index(drop=True)
            df_false = df[df["foil"] == True].sort_values("pair_id").reset_index(drop=True)
            df_true["foil"] = df_false["caption"].tolist()
            df = df_true[["image_path", "caption", "foil", "data_type"]]
    elif (task == "correlation") or (task == "regression"):
        df = _dataset_preprocess(df, base_path, flickr_flag=False)
    
    if (task == "regression"):
        df.loc[:,"human_scores"] = df["human_scores"].apply(lambda x: np.mean(x, dtype=float))
    

    return df, task, wino_tag


def get_sentencepiece_model_for_beit3(sentencepiece_model):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(sentencepiece_model)


def build_transform(input_size=480):
    """Resize to a square, convert to a tensor and normalise. Training uses the same
    transform as evaluation (no augmentation), for both metrics."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])


def scale_scores(scores, benchmark, mode="nominal"):
    """Map human scores to [0, 1].

    mode="nominal" uses the benchmark's rating scale (NOMINAL_SCALES), falling back to
    the observed range for unlisted benchmarks; mode="observed" uses the observed
    minimum and maximum, as the published BEiTScore++ numbers did. Both are affine and
    monotone, so rank correlations do not change; interval coverage can.
    """
    flat = np.array([v for s in scores for v in (s if isinstance(s, list) else [s])], dtype=float)
    lo_obs, hi_obs = float(flat.min()), float(flat.max())
    benchmark = canonical_benchmark(benchmark)
    if mode == "nominal" and benchmark in NOMINAL_SCALES:
        lo, hi = NOMINAL_SCALES[benchmark]
        if lo_obs < lo or hi_obs > hi:
            raise ValueError(f"{benchmark}: observed scores [{lo_obs}, {hi_obs}] fall outside the nominal scale [{lo}, {hi}]")
        if (lo_obs, hi_obs) != (lo, hi):
            print(f"{benchmark}: observed range [{lo_obs}, {hi_obs}] differs from the nominal [{lo}, {hi}]; using the nominal one")
    elif mode in ("nominal", "observed"):
        if mode == "nominal":
            print(f"{benchmark}: no nominal scale listed, using the observed range [{lo_obs}, {hi_obs}]")
        lo, hi = lo_obs, hi_obs
    else:
        raise ValueError(f"Unknown normalisation mode: {mode}")
    span = (hi - lo) or 1.0
    return [((np.asarray(s, dtype=float) - lo) / span).tolist() if isinstance(s, list) else (float(s) - lo) / span
            for s in scores]


class BEiTDataset(torch.utils.data.Dataset):
    def __init__(
        self, benchmark,base_path,tokenizer, num_max_bpe_tokens=256,split="test", min_max_scale = False, use_clusters=False, _task = "", seed = 0
    ):
        """min_max_scale: False (raw scores), True or "nominal" (benchmark scale, see
        NOMINAL_SCALES) or "observed" (the split's own min and max, as in the paper)."""
        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.base_path = base_path
        self.benchmark = benchmark
        self.loader = default_loader
        self.transform = build_transform()
        self.split = split
        self.use_clusters = use_clusters
        df,task,wino_tag = get_data(benchmark, base_path, split, seed=seed)
        if _task != "":
            task = _task
        self.task = task
        self.wino_tag = wino_tag
        if (not wino_tag) and (task == "foil"):
            df_true = df[["image_path","caption","data_type"]].copy()
            df_true["pair_id"] = df_true.index.astype(str)
            df_true["foil"] = False
            df_false = df[["image_path","foil","data_type"]].copy()
            df_false["pair_id"] = df_false.index.astype(str)
            df_false.rename(columns={"foil": "caption"}, inplace=True)
            df_false["foil"] = True
            df = pd.concat([df_true,df_false], axis=0).reset_index(drop=True)
        self.items = df
        if (task == "foil") or (task == "correlation"):
            self.items["candidates"] = self.items["caption"].to_list()
            if task == "foil":
                self.items["is_correct_caption"] = self.items["foil"].apply(lambda x: x == False).tolist()
            if (task == "foil"):
                self.gt_scores = self.items["foil"].to_list()
            else:
                self.gt_scores = self.items["human_scores"].to_list()
                if min_max_scale:
                    mode = "nominal" if min_max_scale is True else min_max_scale
                    self.gt_scores = scale_scores(self.gt_scores, benchmark, mode=mode)
        
        if self.use_clusters & ("cluster" in self.items.columns):
            self.clusters = self.items["cluster"].to_list()
            self.use_clusters = True
        else:
            self.use_clusters = False

    def _get_image(self, image_path: str):

        image = self.loader(image_path)
        return self.transform(image)
    
    def _get_text_segment(self, text_segment, max_len=None):
        """Tokenize the text segment and prepare padding."""
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            tokens = text_segment[:]

        if len(tokens) == 0:
            raise RuntimeError("The text segment should contain at least one token!")

        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        # Add BOS and EOS tokens
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        num_tokens = len(tokens)

        # Create padding mask: 0 for tokens, 1 for padding
        padding_mask = [0] * num_tokens

        return tokens, padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        """Load an image-text pair and assign it to the data dict."""
        item = self.items.iloc[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["candidates"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)

        data["language_tokens"] = torch.tensor(language_tokens)
        data["padding_mask"] = torch.tensor(padding_mask)
        if self.task == "foil":
            data["targets"] = torch.tensor(item["is_correct_caption"], dtype=torch.float32)   
        else:
            data["targets"] = torch.tensor(item["human_scores"], dtype=torch.float32)        
    
    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data
    
    def __getscores__(self):
        """Optional: Return human scores if available."""
        return self.items["human_scores"].to_list()
    
    def __getminmaxscores__(self):
        return self.items["human_scores"].explode().min(), self.items["human_scores"].explode().max()
    
    def __len__(self) -> int:
        return len(self.items)


class RegressionDataset(BEiTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_text_segment(self, text_segment, max_len=None):
        """Tokenize the text segment and prepare padding."""
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            tokens = text_segment[:]

        if len(tokens) == 0:
            raise RuntimeError("The text segment should contain at least one token!")

        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        # Add BOS and EOS tokens
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        num_tokens = len(tokens)

        # Create padding mask: 0 for tokens, 1 for padding
        padding_mask = [0] * num_tokens

        return tokens, padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        """Load an image-text pair and assign it to the data dict."""
        item = self.items.iloc[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["candidates"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)

        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask
        if self.task == "foil":
            data["targets"] = torch.tensor(item["is_correct_caption"], dtype=torch.float32)   
        else:
            data["targets"] = torch.tensor(item["human_scores"], dtype=torch.float32)        
        
    def __getscores__(self):
        """Optional: Return human scores if available."""
        return self.items["human_scores"].to_list()
    
    def __getminmaxscores__(self):
        return self.items["human_scores"].explode().min(), self.items["human_scores"].explode().max()


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem):
    
    if dataset.use_clusters:
        print("Start Cluster Sampler")
        sampler = ClusterBatchSampler(dataset, batch_size=batch_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            collate_fn= merge_batch_tensors_by_dict_key,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if is_train else False ,  # controlled here
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=is_train,
            collate_fn= merge_batch_tensors_by_dict_key,
        )


def create_dataset_by_split(args, split, is_train=True):
    
    if args.task == "stage1":
        from ..stage1.dataset import BCEDataset
        dataset_class = BCEDataset
    else:
        dataset_class = RegressionDataset

    tokenizer = get_sentencepiece_model_for_beit3(args.sentencepiece_model)

    opt_kwargs = {}
    if args.task == "stage1":
        opt_kwargs["use_original_foil"] = getattr(args, "stage1_original_foil", False)
    
    dataset = dataset_class(benchmark=args.benchmark,base_path=args.data_path, tokenizer=tokenizer,split=split, **opt_kwargs, use_clusters=args.use_clusters, seed=getattr(args, "seed", 0))
    
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = args.batch_size

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size, 
        num_workers=args.num_workers, pin_mem=args.pin_mem,
    )


def create_downstream_dataset(args, is_eval=False, mode = ""):
    if is_eval:
        return create_dataset_by_split(args, split="test" + mode, is_train=False)
    else:
        data_loader_train = create_dataset_by_split(args, split="train", is_train=True)
        data_loader_val = create_dataset_by_split(args, split="val", is_train=False)
        return {"data_loader_train":data_loader_train, "data_loader_val":data_loader_val}
