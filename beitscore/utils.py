# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified for BEiTScore++.
# --------------------------------------------------------

import datetime
import io
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from timm.utils import get_state_dict

from .paths import FINETUNE_ROOT, MODELS_ROOT


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.rank = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_steps=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        # max_steps, when given, overrides the length of the iterable
        total_len = max_steps if max_steps is not None else len(iterable)
        space_fmt = ':' + str(len(str(total_len))) + 'd'

        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if (i % print_freq == 0 or i == total_len - 1) and self.rank == 0:
                eta_seconds = iter_time.global_avg * (total_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            if max_steps is not None and i >= max_steps:
                break
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.rank == 0:
            print(f'{header} Total time: {total_time_str} ({total_time / total_len:.4f} s / it)')


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1, sched_type="cos"):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
        
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if sched_type == "cos":
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    elif sched_type == "linear":
        schedule = np.linspace(base_value, final_value, epochs * niter_per_ep - warmup_iters)
    else:
        raise NotImplementedError()

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    checkpoint_path = Path(args.output_dir) / ('checkpoint-%s.pth' % epoch)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    if model_ema is not None:
        to_save['model_ema'] = get_state_dict(model_ema)
    torch.save(to_save, checkpoint_path)


def auto_load_resume_model(args, resume, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    if not resume:
        return
    
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # checkpoints saved from DistributedDataParallel prefix every key with 'module.'
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model_without_ddp.load_state_dict(state_dict)
    print(f"Loaded model from checkpoint: {resume}")

    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = 0

        if hasattr(args, 'model_ema') and args.model_ema and 'model_ema' in checkpoint:
            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])

        print("Loaded optimizer, EMA, and scaler state.")


# The implementation code is modified from open_clip (https://github.com/mlfoundations/open_clip.git)
def update_log_and_save_dir(args):
    args.base_dir = args.output_dir = args.log_dir = FINETUNE_ROOT
    os.makedirs(args.base_dir, exist_ok=True)

    args.models_dir = MODELS_ROOT
    os.makedirs(args.models_dir, exist_ok=True)
    
    base_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(base_dir, exist_ok=True)

    # Find existing run-* directories
    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.fullmatch(r"run-\d+", d)
    ]

    run_nums = sorted([int(re.search(r"run-(\d+)", d).group(1)) for d in run_dirs])

    if run_nums:
        last_run_num = run_nums[-1]
        last_run_path = os.path.join(base_dir, f"run-{last_run_num}")
        if not os.listdir(last_run_path):  # Directory is empty
            new_path = last_run_path
        else:
            new_path = os.path.join(base_dir, f"run-{last_run_num + 1}")
            os.makedirs(new_path)
    else:
        new_path = os.path.join(base_dir, "run-0")
        os.makedirs(new_path)

    args.output_dir = new_path
    args.log_dir = new_path

    # Save arguments to a text file
    args_file = os.path.join(new_path, "arguments_used.txt")
    with open(args_file, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


def get_resume_and_finetune_checkpoint(args):
    if not(args.get_checkpoint_from_finetuning):
        args.annotation_data_path = None
        if args.model_size == "base":
            if args.model_type == "none":
                finetune = "{}/beit3_base_patch16_480_vqa.pth".format(args.models_dir)
            elif args.model_type == "indomain":
                finetune = "{}/beit3_base_indomain_patch16_480_vqa.pth".format(args.models_dir)
            elif args.model_type == "nlvr2":
                finetune = "{}/beit3_base_indomain_patch16_224_nlvr2.pth".format(args.models_dir)
            else:
                raise ValueError(f"Unsupported model_type '{args.model_type}' for model_size 'base' in task 'vqaregression'")
        elif args.model_size == "large":
            if args.model_type == "none":
                finetune = "{}/beit3_large_patch16_480_vqa.pth".format(args.models_dir)
            elif args.model_type == "indomain":
                finetune = "{}/beit3_large_indomain_patch16_480_vqa.pth".format(args.models_dir)
            elif args.model_type == "nlvr2":
                finetune = "{}/beit3_large_indomain_patch16_224_nlvr2.pth".format(args.models_dir)
            else:
                raise ValueError(f"Unsupported model_type '{args.model_type}' for model_size 'large' in task 'vqaregression'")
        else:
            raise ValueError(f"Unsupported model_size '{args.model_size}'")

    else:
        checkpoint_number = args.checkpoint_number
        if args.checkpoint_number == -1:
            checkpoint_number = "best"

        finetune = "{}/{}/run-{}/checkpoint-{}.pth".format(args.base_dir, args.task_resume, args.use_run, checkpoint_number) 
        
    
    return finetune
