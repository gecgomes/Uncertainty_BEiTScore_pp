"""Download the released models and data from the Hugging Face Hub into the folders the
code reads ($BEITSCORE_MODELS and $BEITSCORE_DATA, see configs/paths.example.sh).

    source configs/paths.sh
    python scripts/download_release.py models                     # BEiTScore and BEiTScore++, base and large
    python scripts/download_release.py longcapvlcp                # the LongCapVLCP benchmark
    python scripts/download_release.py stage1                     # Flickr30k comes from Kaggle automatically
    python scripts/download_release.py stage2                     # Stage-2 training and calibration data
    python scripts/download_release.py all

What goes where:
  models       <namespace>/beitscore   -> $BEITSCORE_MODELS/beitscore/<size>/   (+ beit3.spm)
               <namespace>/beitscorepp -> $BEITSCORE_MODELS/beitscorepp/<size>/ (weights, config, calibration)
  longcapvlcp  <namespace>/LongCapVLCP -> $BEITSCORE_DATA/release/longcapvlcp/, then its images are
               extracted and $BEITSCORE_DATA/longcapvlcp/longcapvlcp.pkl is written (what evaluate_errors.py reads)
  stage1       <namespace>/beitscore-stage1 -> $BEITSCORE_DATA/release/stage1/, then the images are
               extracted (PixelProse) or downloaded (COCO, Open Images, TextCaps) and checked, Flickr30k
               is downloaded from Kaggle into $BEITSCORE_DATA/flickr30k/Images (or taken from
               --flickr30k), and
               $BEITSCORE_DATA/Stage1/stage1.pkl is written (what scripts/train.py reads)
  stage2       <namespace>/beitscore-stage2 -> $BEITSCORE_DATA/release/stage2/, then the images are
               extracted (Polaris, Nebula, EvalMuse-40K) or downloaded (COCO) and checked, Flickr8K
               comes from its archive, and $BEITSCORE_DATA/stage2/stage2.pkl is written (what
               scripts/train.py --benchmark regression|quant and scripts/beitscore_pp/calibrate.py read)

The Stage-1 images take about 150 GB on disk (75 GB downloaded from the Hub, the rest from
the original sources). Rerunning resumes: finished files are not downloaded again.
"""
import argparse
import os
import sys

from beitscore.paths import DATA_ROOT, MODELS_ROOT

NAMESPACE = os.environ.get("BEITSCORE_HF_NAMESPACE", "goncaloecgomes")
REPOS = {
    "beitscore": ("beitscore", "model"),
    "beitscorepp": ("beitscorepp", "model"),
    "longcapvlcp": ("LongCapVLCP", "dataset"),
    "stage1": ("beitscore-stage1", "dataset"),
    "stage2": ("beitscore-stage2", "dataset"),
}


def snapshot(key, local_dir, namespace, allow_patterns=None):
    from huggingface_hub import snapshot_download

    name, repo_type = REPOS[key]
    repo_id = f"{namespace}/{name}"
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    print(f"Downloading {repo_id} ({repo_type}) to {local_dir}")
    try:
        return snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir,
                                 allow_patterns=allow_patterns)
    except (RepositoryNotFoundError, GatedRepoError):
        print(f"{repo_id} is not available: it is not released yet, or it is private and HF_TOKEN does not "
              f"give access to it. Skipped.", file=sys.stderr)
        return None


def get_images(release_dir, out_dir, out_file, table, archives, workers):
    """Run scripts/get_images.py on a downloaded release."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_images.py")
    spec = importlib.util.spec_from_file_location("get_images", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--release", release_dir, "--out_dir", out_dir, "--out_file", out_file, "--table", table,
            "--workers", str(workers)]
    for source, folder in archives.items():
        argv += ["--archive", f"{source}={folder}"]
    return module.main(module.get_args(argv))


def download_models(args):
    sizes = [f"{s}/*" for s in args.sizes]
    snapshot("beitscore", os.path.join(MODELS_ROOT, "beitscore"), args.namespace,
             allow_patterns=sizes + ["beit3.spm", "*.md"])
    spm = os.path.join(MODELS_ROOT, "beitscore", "beit3.spm")
    if os.path.exists(spm) and not os.path.exists(os.path.join(MODELS_ROOT, "beit3.spm")):
        os.symlink(spm, os.path.join(MODELS_ROOT, "beit3.spm"))
    snapshot("beitscorepp", os.path.join(MODELS_ROOT, "beitscorepp"), args.namespace,
             allow_patterns=sizes + ["*.md"])


def download_longcapvlcp(args):
    release = os.path.join(DATA_ROOT, "release", "longcapvlcp")
    if snapshot("longcapvlcp", release, args.namespace) is None:
        return
    get_images(release, os.path.join(DATA_ROOT, "longcapvlcp", "images"),
               os.path.join(DATA_ROOT, "longcapvlcp", "longcapvlcp.pkl"), "longcapvlcp", {}, args.workers)


FLICKR30K_HANDLE = "adityajn105/flickr30k"
FLICKR30K_URL = f"https://www.kaggle.com/datasets/{FLICKR30K_HANDLE}"
FLICKR30K_DIR = os.path.join(DATA_ROOT, "flickr30k", "Images")


def _find_images_folder(root):
    """The folder of the Kaggle archive that holds the .jpg files (Images/ in the current version)."""
    best, most = None, 0
    for folder, _, files in os.walk(root):
        n = sum(f.lower().endswith(".jpg") for f in files)
        if n > most:
            best, most = folder, n
    return best


def get_flickr30k(target=FLICKR30K_DIR):
    """$BEITSCORE_DATA/flickr30k/Images, downloaded from Kaggle with kagglehub when it is not there.
    Public Kaggle datasets download without an account; if Kaggle asks for one, log in with
    `kagglehub.login()` or put kaggle.json in ~/.kaggle/. Returns None when it cannot be obtained."""
    import shutil

    if os.path.isdir(target) and any(f.lower().endswith(".jpg") for f in os.listdir(target)):
        print(f"Flickr30k images found in {target}")
        return target
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed (pip install kagglehub); cannot download Flickr30k", file=sys.stderr)
        return None
    print(f"Downloading Flickr30k from {FLICKR30K_URL} (about 4 GB)")
    try:
        images = _find_images_folder(kagglehub.dataset_download(FLICKR30K_HANDLE))
        if images is None:  # the cached copy was already moved out by an earlier run
            images = _find_images_folder(kagglehub.dataset_download(FLICKR30K_HANDLE, force_download=True))
    except Exception as exc:  # noqa: BLE001  (network, authentication, ...)
        print(f"Flickr30k download failed: {exc}", file=sys.stderr)
        return None
    cache = "the Kaggle cache"
    if images is None:
        print(f"No .jpg files found in the Kaggle download at {cache}", file=sys.stderr)
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.lexists(target):
        os.remove(target) if os.path.islink(target) else shutil.rmtree(target)
    shutil.move(images, target)  # out of the kagglehub cache, so the 4 GB are not kept twice
    print(f"Flickr30k images moved to {target}")
    return target


def download_stage1(args):
    release = os.path.join(DATA_ROOT, "release", "stage1")
    if snapshot("stage1", release, args.namespace) is None:
        return
    flickr = args.flickr30k or (None if args.no_kaggle else get_flickr30k())
    archives = {"flickr30k": flickr} if flickr else {}
    if not archives:
        print(f"WARNING: no Flickr30k folder; Flickr30k images will be missing. Download {FLICKR30K_URL}, unzip it "
              f"and pass its Images/ folder with --flickr30k (or put it in {FLICKR30K_DIR}).")
    os.makedirs(os.path.join(DATA_ROOT, "Stage1"), exist_ok=True)
    get_images(release, os.path.join(DATA_ROOT, "Stage1", "images"), os.path.join(DATA_ROOT, "Stage1", "stage1.pkl"),
               "stage1", archives, args.workers)


FLICKR8K_DRIVE_ID = "1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn"  # Flickr8k_Dataset.zip, the archive CLIPScore uses
FLICKR8K_DIR = os.path.join(DATA_ROOT, "flickr8k", "Flicker8k_Dataset")


def get_flickr8k(target=FLICKR8K_DIR):
    """$BEITSCORE_DATA/flickr8k/Flicker8k_Dataset, from the Flickr8K archive of CLIPScore's
    flickr8k_example/download.py (Google Drive) when it is not there. Returns None when it
    cannot be obtained."""
    import shutil
    import zipfile

    if os.path.isdir(target) and any(f.lower().endswith(".jpg") for f in os.listdir(target)):
        print(f"Flickr8K images found in {target}")
        return target
    try:
        import gdown
    except ImportError:
        print("gdown is not installed (pip install gdown); cannot download Flickr8K", file=sys.stderr)
        return None
    archive = os.path.join(os.path.dirname(target), "Flickr8k_Dataset.zip")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print("Downloading Flickr8K (about 1 GB)")
    if not os.path.exists(archive) and gdown.download(id=FLICKR8K_DRIVE_ID, output=archive, quiet=False) is None:
        print("Flickr8K download failed", file=sys.stderr)
        return None
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir() and "__MACOSX" not in m.filename
                   and m.filename.lower().endswith(".jpg")]
        os.makedirs(target, exist_ok=True)
        for m in members:
            with z.open(m) as src, open(os.path.join(target, os.path.basename(m.filename)), "wb") as dst:
                shutil.copyfileobj(src, dst)
    os.remove(archive)
    print(f"Flickr8K images in {target}")
    return target


def download_stage2(args):
    release = os.path.join(DATA_ROOT, "release", "stage2")
    if snapshot("stage2", release, args.namespace) is None:
        return
    flickr = args.flickr8k or get_flickr8k()
    archives = {"flickr8k": flickr} if flickr else {}
    if not archives:
        print(f"WARNING: no Flickr8K folder; its images will be missing. Pass the folder with the Flickr8K .jpg "
              f"files with --flickr8k (or put it in {FLICKR8K_DIR}).")
    os.makedirs(os.path.join(DATA_ROOT, "stage2"), exist_ok=True)
    get_images(release, os.path.join(DATA_ROOT, "stage2", "images"), os.path.join(DATA_ROOT, "stage2", "stage2.pkl"),
               "stage2", archives, args.workers)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", nargs="+", choices=["models", "longcapvlcp", "stage1", "stage2", "all"])
    parser.add_argument("--sizes", nargs="+", default=["base", "large"], choices=["base", "large"])
    parser.add_argument("--flickr30k", default=None,
                        help="Folder with the Flickr30k .jpg files (for stage1). Default: $BEITSCORE_DATA/flickr30k/Images, "
                             "downloaded from Kaggle when it is not there")
    parser.add_argument("--no_kaggle", action="store_true", help="Do not download Flickr30k from Kaggle")
    parser.add_argument("--flickr8k", default=None,
                        help="Folder with the Flickr8K .jpg files (for stage2). Default: $BEITSCORE_DATA/flickr8k/"
                             "Flicker8k_Dataset, downloaded when it is not there")
    parser.add_argument("--namespace", default=NAMESPACE,
                        help="Hugging Face user or organisation (default: $BEITSCORE_HF_NAMESPACE or %(default)s)")
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args(argv)


def main(args):
    what = {"models", "longcapvlcp", "stage1", "stage2"} if "all" in args.what else set(args.what)
    if not os.environ.get("BEITSCORE_DATA") or not os.environ.get("BEITSCORE_MODELS"):
        print("WARNING: BEITSCORE_DATA or BEITSCORE_MODELS is not set; run `source configs/paths.sh` first",
              file=sys.stderr)
    if "models" in what:
        download_models(args)
    if "longcapvlcp" in what:
        download_longcapvlcp(args)
    if "stage1" in what:
        download_stage1(args)
    if "stage2" in what:
        download_stage2(args)


if __name__ == "__main__":
    main(get_args())
