# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified for BEiTScore++.
# --------------------------------------------------------

import torch


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
    return missing_keys, unexpected_keys


def read_checkpoint(ckpt_path, model_key="model|module"):
    """Return the state dict stored in a checkpoint file or URL.

    `model_key` lists the keys to try ("model|module"); if none is present the file
    itself is taken as the state dict. Released checkpoints (S-03) are plain state
    dicts, in .pth or .safetensors.
    """
    if ckpt_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(ckpt_path, map_location="cpu", check_hash=True)
    elif ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Load ckpt from %s" % ckpt_path)
    for key in model_key.split("|"):
        if key in checkpoint:
            print("Load state_dict by model_key = %s" % key)
            return checkpoint[key]
    return checkpoint


# The implementation code is modified from DeiT (https://github.com/facebookresearch/deit.git)
def interpolate_pos_embed(model, checkpoint_model):
    """Resize the checkpoint's position embeddings to the model's patch grid, in place."""
    for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
        if pos_embed_key not in checkpoint_model:
            continue
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
            # being consistent with Fairseq, which starts from 2 for position embedding
            torchscale_model = True
            num_patches = model.beit3.vision_embed.num_patches
            num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
        else:
            torchscale_model = False
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
        # height (== width) of the checkpoint and of the model grid
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size == new_size:
            continue
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        if torchscale_model:
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        # only the position tokens are interpolated; class and extra tokens are kept
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        if torchscale_model:
            new_pos_embed = new_pos_embed.squeeze(0)
        checkpoint_model[pos_embed_key] = new_pos_embed


def load_model_and_may_interpolate_beitscore(ckpt_path, model, model_key, model_prefix, strict=False):
    """Load any finished checkpoint (BEiTScore, BEiTScore++, BEiT-3 upstream).

    Head tensors whose shape does not match the model are dropped, except a
    single-output head loaded into a three-output head, which is tiled.
    strict=True (used for evaluation and calibration) raises instead when a weight of
    the scoring head is left at its random initialisation or a head tensor is dropped, so
    that a BEiTScore checkpoint cannot be scored as a BEiTScore++ model by mistake.
    """
    checkpoint_model = read_checkpoint(ckpt_path, model_key)
    state_dict = model.state_dict()
    for k in list(checkpoint_model.keys()):
        if k.startswith('head.') and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            ckpt_tensor = checkpoint_model[k]
            model_tensor = state_dict[k]
            if ckpt_tensor.shape[0] == 1 and model_tensor.shape[0] == 3:
                print(f"Expanding key {k} from shape {ckpt_tensor.shape} to {model_tensor.shape} by tiling")
                checkpoint_model[k] = ckpt_tensor.repeat(3, *([1] * (ckpt_tensor.dim() - 1)))
            else:
                if strict:
                    raise ValueError(f"{ckpt_path}: head tensor {k} has shape {tuple(ckpt_tensor.shape)}, the model "
                                     f"expects {tuple(model_tensor.shape)}; the checkpoint does not match this model")
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    missing, _ = load_state_dict(model, checkpoint_model, prefix=model_prefix)
    head_missing = [k for k in missing if k.startswith(("head.", "trunk.")) or "_quantile_pool" in k]
    if strict and head_missing:
        raise ValueError(f"{ckpt_path}: {len(head_missing)} weights of the scoring head are not in the checkpoint "
                         f"(e.g. {head_missing[:4]}), so they would stay random. Is it a BEiTScore checkpoint "
                         "loaded as BEiTScore++ (or the reverse)?")


def quantile_bias_offsets(confidence_levels, max_bias):
    """Initial offsets of the lower and upper quantile heads relative to the median head.

    The offsets follow the standard normal quantiles of the interval ends, scaled so
    that the outermost one has magnitude |max_bias|:

        offset(q) = Phi^-1(q) / max_j |Phi^-1(q_j)| * max_bias

    With max_bias = -2.5 and confidence levels [0.9, 0.85, 0.8, 0.75, 0.7], the lower
    heads start between +1.58 and +2.5 above the median head's bias and the upper heads
    between -1.58 and -2.5 below it. In the chained head the lower bounds are
    sigmoid(z) * median and the upper bounds median + sigmoid(z) * (1 - median), so
    these offsets make the initial intervals narrow; they widen during training. This
    is the initialisation used for the released BEiTScore++ checkpoints.
    """
    from scipy.stats import norm

    low_q = sorted((1 - p) / 2 for p in confidence_levels)
    high_q = sorted(1 - (1 - p) / 2 for p in confidence_levels)
    low_probit = torch.tensor([norm.ppf(q) for q in low_q], dtype=torch.float32)
    high_probit = torch.tensor([norm.ppf(q) for q in high_q], dtype=torch.float32)
    low = torch.sort(low_probit / low_probit.abs().max() * max_bias).values
    high = high_probit / high_probit.abs().max() * max_bias
    return low, high


def load_model_and_may_interpolate_beitscorepp(args, ckpt_path, model, model_key, model_prefix):
    """Initialise a BEiTScore++ model from a BEiTScore checkpoint.

    The BEiTScore head (head.0 to head.3) becomes the trunk and the median head; the
    lower and upper heads copy the median head's weights, with the bias offsets of
    `quantile_bias_offsets(model.quantiles, args.bias)`.
    """
    checkpoint_model = read_checkpoint(ckpt_path, model_key)
    remapped_state = {}
    state_dict = model.state_dict()
    for k, v in checkpoint_model.items():
        # ---- trunk (old head[0:3]) ----
        if k.startswith("head.0") or k.startswith("head.1") or k.startswith("head.2"):
            remapped_state[k.replace("head.", "trunk.")] = v
        # ---- median head (old head[3]), copied to the lower and upper heads ----
        elif k.startswith("head.3.") or k == "head.3":
            remapped_state[k.replace("head.3", "mid_quantile_pool", 1)] = v
            low_k = k.replace("head.3", "low_quantile_pool", 1)
            high_k = k.replace("head.3", "high_quantile_pool", 1)
            if k.endswith(".weight"):
                remapped_state[low_k] = v.repeat(state_dict[low_k].shape[0], 1)
                remapped_state[high_k] = v.repeat(state_dict[high_k].shape[0], 1)
            elif k.endswith(".bias"):
                low_biases, high_biases = quantile_bias_offsets(list(model.quantiles), args.bias)
                assert len(low_biases) == state_dict[low_k].shape[0], "one lower head per confidence level"
                assert len(high_biases) == state_dict[high_k].shape[0], "one upper head per confidence level"
                print("Low Biases:", low_biases)
                print("High Biases:", high_biases)
                scalar = v[0]
                remapped_state[low_k] = scalar + low_biases.to(v.dtype)
                remapped_state[high_k] = scalar + high_biases.to(v.dtype)
        # ---- drop the rest of the old head ----
        elif k.startswith("head."):
            print(f"Dropping old head key: {k}")
        else:
            remapped_state[k] = v

    interpolate_pos_embed(model, remapped_state)
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
