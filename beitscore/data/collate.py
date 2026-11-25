import torch
from torch.nn.utils.rnn import pad_sequence


def merge_batch_tensors_by_dict_key(batch):
    flattened = []

    for sample in batch:
        if any(isinstance(v, tuple) for v in sample.values()):
            n = len(next(v for v in sample.values() if isinstance(v, tuple)))
            for i in range(n):
                new_sample = {}
                for key, value in sample.items():
                    if isinstance(value, tuple):
                        new_sample[key] = value[i]      # candidate i
                    else:
                        new_sample[key] = value         # shared field
                flattened.append(new_sample)

        else:
            flattened.append(sample)

    batch = flattened

    batch_tensors = {}
    for tensor_key in batch[0]:
        values = [d[tensor_key] for d in batch]
        if "targets" in tensor_key:
            if any(len(v.shape) > 0 and v.shape[0] != values[0].shape[0] for v in values):
                # Variable-length, pad required
                padded_targets = pad_sequence(values, batch_first=True, padding_value=-999)
                batch_tensors["targets"] = padded_targets

                # Create mask for non-padded elements
                target_mask = torch.stack([
                    torch.tensor([True] * len(v) + [False] * (padded_targets.size(1) - len(v)))
                    for v in values
                ])
                batch_tensors["targets_mask"] = target_mask
            else:
                # Same shape, no padding needed
                batch_tensors[tensor_key] = torch.stack(values)
                batch_tensors["targets_mask"] = torch.ones_like(batch_tensors["targets"], dtype=torch.bool)
        elif tensor_key in ["padding_mask","padding_mask_corr","padding_mask_foil"]:
            values = [torch.as_tensor(v) for v in values]  # <-- ensure tensors
            padded_targets = pad_sequence(values, batch_first=True, padding_value=1)
            batch_tensors[tensor_key] = padded_targets
        elif tensor_key in ["language_tokens","language_tokens_corr","language_tokens_foil"]:
            values = [torch.as_tensor(v) for v in values]
            padded_targets = pad_sequence(values, batch_first=True, padding_value=1)
            batch_tensors[tensor_key] = padded_targets
        elif isinstance(values[0], torch.Tensor):
            try:
                batch_tensors[tensor_key] = torch.stack(values)
            except RuntimeError as e:
                raise RuntimeError(f"Error stacking key '{tensor_key}': {e}")
        else:
            batch_tensors[tensor_key] = torch.tensor(values, dtype=torch.long)

    return batch_tensors
