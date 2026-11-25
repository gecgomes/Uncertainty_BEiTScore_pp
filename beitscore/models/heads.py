# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified for BEiTScore++.
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.registry import register_model

from .beit3_wrapper import BEiT3Wrapper, _get_base_config, _get_large_config


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:,0,:]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForRegression(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            large = False,
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForRegression, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 216), 
            norm_layer(216), 
            nn.GELU(),  
            nn.Linear(216, num_classes), 
        )
        self.head.apply(self._init_weights)
        self.last_activation = nn.Sigmoid()
        self.max_value, self.min_value = 1, 0
    def forward(self, image, text_description, padding_mask, only_inference = False, **kwargs):
        outputs = self.beit3(
            textual_tokens=text_description, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]

        cls_rep = self.pooler(x)
        logits = self.head(cls_rep)
        y = self.last_activation(logits)
        if only_inference:
            return y*(self.max_value - self.min_value) + self.min_value
        else:
            return y, logits

class BEiT3ForQuantileRegression(BEiT3Wrapper):
    def __init__(self, args, quantiles=[0.9,0.85,0.8,0.75,0.7], large=False, norm_layer=nn.LayerNorm, chained_construction=False, **kwargs):
        super().__init__(args=args)
        self.chained_construction = chained_construction
        embed_dim = args.encoder_embed_dim

        self.pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )

        # shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, 216),
            norm_layer(216),
            nn.GELU(),
        )
        self.quantiles = quantiles
        # quantile heads
        self.mid_quantile_pool = nn.Linear(216, 1)
        self.low_quantile_pool = nn.Linear(216, len(quantiles))
        self.high_quantile_pool = nn.Linear(216, len(quantiles))

        # init trunk + mid from pretrained later
        self.trunk.apply(self._init_weights)
        self.mid_quantile_pool.apply(self._init_weights)
        self.low_quantile_pool.apply(self._init_weights)
        self.high_quantile_pool.apply(self._init_weights)

        self.last_activation = nn.Sigmoid()

    def forward(self, image, text_description, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=text_description,
            visual_tokens=image,
            text_padding_position=padding_mask,
        )

        x = outputs["encoder_out"]

        cls_rep = self.pooler(x)

        features = self.trunk(cls_rep)


        if self.chained_construction:
            delta_low = self.last_activation(self.low_quantile_pool(features)).squeeze(-1)
            q_mid = self.last_activation(self.mid_quantile_pool(features)).squeeze(-1)
            delta_high = self.last_activation(self.high_quantile_pool(features)).squeeze(-1)

            ci = [q_mid] 
            if len(self.quantiles) > 1:
                for i in range(0, delta_low.shape[1]):
                    ci.insert(0,delta_low[:, -(i+1)]*(ci[0])) 
                for i in range(0, delta_high.shape[1]):
                    ci.append(ci[-1] + delta_high[:, i]*(1-ci[-1]))
            else:
                ci.insert(0,delta_low*(ci[0]))
                ci.append(ci[-1] + delta_high*(1-ci[-1]))

            y = torch.stack(ci, dim=-1)
        else:
            q_low = self.last_activation(self.low_quantile_pool(features)).squeeze(-1)
            q_mid = self.last_activation(self.mid_quantile_pool(features)).squeeze(-1)
            q_high = self.last_activation(self.high_quantile_pool(features)).squeeze(-1)

            y = torch.cat([q_low, q_mid.unsqueeze(-1), q_high], dim=-1)
        return y


@register_model
def beit3_base_patch16_480_stage1(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForRegression(args, num_classes=1, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_stage1(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForRegression(args, num_classes=1, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_regression(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForRegression(args, num_classes=1, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_regression(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForRegression(args, num_classes=1, **kwargs)
    return model


@register_model
def beit3_base_patch16_480_quant(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForQuantileRegression(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_quant(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForQuantileRegression(args, **kwargs)
    return model
