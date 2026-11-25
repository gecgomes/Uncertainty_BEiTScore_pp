# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified for BEiTScore++.
# --------------------------------------------------------

import datetime
import json
import math
import os
import sys
import time
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from timm.utils import ModelEma

from . import metrics, utils


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class RegressorHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.scores = []
        self.human_scores = []
        self.criterion = nn.L1Loss(reduction='mean')
        
    def train_batch(self, model, image, language_tokens, padding_mask, targets, targets_mask):
        logits,_ = model(
            image=image, text_description=language_tokens, 
            padding_mask=padding_mask)
        loss = self.criterion(input=logits.squeeze(1).float(), target=targets.float())
        targets.detach().cpu()
        return {"loss": loss}
            
    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split
        self.scores = []
        self.human_scores = []

    def eval_batch(self, model, image, language_tokens, padding_mask, targets, targets_mask):
        logits = model(image=image, text_description=language_tokens, padding_mask=padding_mask, only_inference = True)
      
        self.scores.append(logits[:,0])
        decoded_targets = []
        for target_row, mask_row in zip(targets, targets_mask):
            # Keep only the values where mask is True
            valid_values = target_row[mask_row].tolist()
            decoded_targets.append(valid_values)
        self.human_scores += decoded_targets

    def after_eval(self, return_results, **kwargs):
        scores = torch.cat(self.scores).cpu().numpy()
        # One row per annotation (aggregation method A of the CLIPScore paper,
        # https://arxiv.org/pdf/2104.08718).
        x, y = metrics.explode_pairs(scores, self.human_scores)
        loss = float(np.mean((x - y) ** 2))  # mean squared error
        corr = metrics.correlations(scores, self.human_scores)
        print("... Human Correlation Scores")
        print('Loss (MSE): ', loss)
        print('Tau-b: {:.3f}'.format(corr["tau_b"]))
        print('Tau-c: {:.3f}'.format(corr["tau_c"]))
        print('Rho: {:.3f}'.format(corr["rho"]))

        if return_results:
            return {"loss": loss, **corr}


def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, 
        optimizer: torch.optim.Optimizer, device: torch.device, 
        handler: TaskHandler, epoch: int, start_steps: int, warmup_steps:int, 
        lr_schedule_values: list, loss_scaler, max_norm: float = 0, 
        update_freq: int = 1, model_ema: Optional[ModelEma] = None, 
        log_writer = None, 
        task = None, mixup_fn=None,max_steps_per_epoch = None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.zero_grad()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header, max_steps=max_steps_per_epoch)):
        if max_steps_per_epoch is not None and data_iter_step >= max_steps_per_epoch:
            break   
        step = data_iter_step // update_freq
        global_step = start_steps + step - warmup_steps  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            results = handler.train_batch(model, **data)
        loss = results["loss"]
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        for k, v in results.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    metric_logger.update(**{k: v.item()})
            elif isinstance(v, (float, int)):
                metric_logger.update(**{k: v})
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_kwargs = {}

            for k, v in results.items():
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        log_kwargs[k] = v.item()
                elif isinstance(v, (float, int)):
                    log_kwargs[k] = v

            log_writer.update(head="train", **log_kwargs)

            kwargs = {
                "loss_scale": loss_scale_value, 
                "lr": max_lr, 
                "min_lr": min_lr, 
                "weight_decay": weight_decay_value, 
                "grad_norm": grad_norm, 
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args,model,optimizer,loss_scaler,task_handler_train, task_handler_val, data_loader_train, data_loader_val,log_writer = None, model_ema = None, device = torch.device("cuda")):
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train.dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.greater_is_better:
        best_results = 0
    else:
        best_results = torch.inf
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    early_stopping_steps = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        if early_stopping_steps == args.early_stopping:
            break
        if log_writer is not None: 
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, task_handler_train, epoch, 
            epoch * num_training_steps_per_epoch,num_training_steps_per_epoch*args.warmup_epochs, lr_schedule_values, loss_scaler, 
            args.clip_grad, args.update_freq, model_ema, log_writer, args.task, mixup_fn = None, max_steps_per_epoch = (args.instances_per_eval // total_batch_size) if args.instances_per_eval > 0 else None
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            results = evaluate(data_loader_val, model, device, task_handler_val)
            print(f"Performance of the network on the {len(data_loader_val.dataset)}: {results[args.task_key]:.1f}%")
            if ((best_results < results[args.task_key]) and (args.greater_is_better)) or ((best_results > results[args.task_key]) and (not args.greater_is_better)) :
                best_results = results[args.task_key]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                early_stopping_steps = 0
            else:
                early_stopping_steps += 1
                
            print(f'Max performance: {best_results:.2f}%')
            if log_writer is not None:
                log_writer.update(acc=results[args.task_key], head="perf", step=epoch)
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in results.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         # **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def evaluate(data_loader, model, device, handler, save_values= False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model = model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            handler.eval_batch(model=model, **data)

    results = handler.after_eval(return_results= True)

    model = model.train(True)
    return results
