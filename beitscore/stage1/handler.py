# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified for BEiTScore++.
# --------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..engine import TaskHandler


class BCEHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.scores = []
        self.target_scores = []
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
    def train_batch(self, model, image, language_tokens, padding_mask, targets, targets_mask):
        _,logits = model(
            image=image, text_description=language_tokens, 
            padding_mask=padding_mask)
        
        loss = self.criterion(input=logits, target=targets)
        return {
            "loss": loss, 
        }
    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split
        self.task = data_loader.dataset.items["task"]
        self.wino_tag = self.task.iloc[0] == "winoground"
        self.multi_task_tag = len(self.task.unique()) > 1
        if self.wino_tag:
            self.pair_ids = data_loader.dataset.items["pair_id"].apply(lambda x: int(x[2:]))
            self.tag_combination = data_loader.dataset.items["pair_id"].apply(lambda x: x[:2])
        else:
            self.pair_ids = data_loader.dataset.items["pair_id"]

        self.scores = []
        self.target_scores = []

    def eval_batch(self, model, image, language_tokens, padding_mask, targets, targets_mask):
        logits = model(image=image, text_description=language_tokens, padding_mask=padding_mask, only_inference = True)
        self.scores.append(logits[:,0])
        self.target_scores.append(targets.squeeze(1))

    def after_eval(self, return_results=True, **kwargs):
        # Concatenate all stored scores and targets
        probs = torch.cat(self.scores).cpu().numpy()
        target_scores = torch.cat(self.target_scores).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        task = self.task.tolist() + self.task.tolist()
        pair_ids = [x for x in self.pair_ids.tolist() for _ in range(2)]
        
        results_df = pd.DataFrame({
            "pair_ids": pair_ids,
            "probs": probs,
            "preds": preds,
            "target_scores": target_scores,
            "task": task
        })
        results_df = results_df[results_df["task"].apply(lambda x: "coreference" not in x.lower())] # filter out coreference tasks of the valse dataset, as it is the only task that lacks caption-like fluency.

        if self.wino_tag:   
            results_df["tag_combination"] = self.tag_combination
        
        grouped = results_df.groupby(["task", "pair_ids"])

        acc_records = []
        for (task, pid), group in grouped:

            if self.wino_tag:
                prob_00 = group[group["tag_combination"] == "00"]["probs"].values[0]
                prob_11 = group[group["tag_combination"] == "11"]["probs"].values[0]
                prob_01 = group[group["tag_combination"] == "01"]["probs"].values[0]
                prob_10 = group[group["tag_combination"] == "10"]["probs"].values[0]
                text_acc = prob_00 > prob_01 and prob_11 > prob_10 # measures whether a model can select the correct caption, given an image.
                image_acc = prob_00 > prob_10 and prob_11 > prob_01 # measures whether a model can select the correct image, given an caption.
                acc = text_acc and image_acc

                acc_records.append({
                    "task": task,
                    "pair_ids": pid,
                    "text_acc": text_acc,
                    "image_acc": image_acc,
                    "acc": acc,
                    "pred": (prob_00, prob_01, prob_10, prob_11)
                })
            else:
                correct_probs = group[group["target_scores"] == 1]["probs"].tolist()
                foil_probs = group[group["target_scores"] == 0]["probs"].tolist()

                correct_prob = correct_probs[0]
                acc = all(correct_prob > fp for fp in foil_probs)

                acc_records.append({
                    "task": task,
                    "pair_ids": pid,
                    "acc": acc
                })

        acc_results = pd.DataFrame(acc_records) 
        # ---- GLOBAL METRICS ----
        recall = recall_score(target_scores, preds)
        precision = precision_score(target_scores, preds)
        f1 = f1_score(target_scores, preds)
        ap = average_precision_score(target_scores, probs)
        auroc = roc_auc_score(target_scores, probs)
        loss = log_loss(target_scores, probs)
        acc_on_order = np.mean(acc_results["acc"])
        print("\n=== GLOBAL Evaluation Results ===")
        print(f"Loss: {loss:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average Precision (AP): {ap:.4f}")
        print(f"AUROC: {auroc:.4f}")
        if self.wino_tag:
            text_acc = np.mean(acc_results["text_acc"])
            image_acc = np.mean(acc_results["image_acc"])
            print(f"Text ACC on Order: {text_acc:.4f}")
            print(f"Image ACC on Order: {image_acc:.4f}")
        print(f"ACC on Order: {acc_on_order:.4f}")

        if self.multi_task_tag:
            # ---- PER TASK METRICS (print only, no return) ----
            print("\n=== Per-Task Evaluation Results ===")
            macro_acc = acc_results.groupby("task")["acc"].mean().mean()
            print(f"MACRO AVERAGE Accuracy: {macro_acc:.4f}")
            for task, df in results_df.groupby("task"):
                r = recall_score(df.target_scores, df.preds)
                p = precision_score(df.target_scores, df.preds)
                f = f1_score(df.target_scores, df.preds)
                ap_task = average_precision_score(df.target_scores, df.probs)
                auroc_task = roc_auc_score(df.target_scores, df.probs)
                loss_task = log_loss(df.target_scores, df.probs)
                acc_on_order = np.mean(acc_results[acc_results["task"] == task]["acc"])
                print(f"\nTask: {task}")
                print(f"  Loss: {loss_task:.4f}")
                print(f"  Recall: {r:.4f}")
                print(f"  Precision: {p:.4f}")
                print(f"  F1 Score: {f:.4f}")
                print(f"  Average Precision (AP): {ap_task:.4f}")
                print(f"  AUROC: {auroc_task:.4f}")
                print(f"  ACC on Order: {acc_on_order:.4f}")

        if return_results:
            return {
                "loss": loss,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "ap": ap,
                "auroc": auroc,
                "acc_on_order":acc_on_order,
            }
