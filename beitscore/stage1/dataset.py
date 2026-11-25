import math
import random

import torch

from ..data.datasets import BEiTDataset


EMPTY = "<empty>"  # stored by the Stage-1 pipeline when a similarity band has no neighbour


def augment_negatives(words,pos_tags, seed=None):
    """Return an incorrect caption made by repeating, swapping or shuffling words.

    Works on copies: the lists stored in the dataset row are never modified.
    """
    if seed is not None:
        random.seed(seed)
    words = list(words)
    pos_tags = list(pos_tags)

    valid_pos = {"NOUN", "NUM", "ADJ", "ADV", "PROPN", "VERB"}

    def repeat_words(words, pos_tags):
        """Repeat a random valid POS word."""
        candidate_idxs = [i for i, p in enumerate(pos_tags) if p in valid_pos]
        if not candidate_idxs:
            return words
        idx = random.choice(candidate_idxs)
        repeat_times = random.choice([2, 8])
        transformed = words
        for _ in range(repeat_times):
            transformed.insert(idx + 1, words[idx])
            pos_tags.insert(idx + 1, pos_tags[idx])  # keep the tags aligned for later steps
        return transformed

    def swap_words(words, pos_tags=None):
        """
        Perform multiple swaps of nearby (or sometimes random) words.
        """
        n = len(words)

        if n <= 3:
            return words  # no real swaps possible

        transformed = words

        # default number of swaps = half the length
        num_swaps = max(1, n // 2)

        for _ in range(num_swaps):

            # 80% of the time: swap nearby words
            if random.random() < 0.8:
                i = random.randint(0, n - 2)
                # swap with next word or the one after it
                j = i + random.choice([1, 2]) if i < n - 2 else i + 1

            # 20% of the time: random positions
            else:
                i, j = random.sample(range(n), 2)

            # perform the swap
            transformed[i], transformed[j] = transformed[j], transformed[i]

        return transformed

    def shuffle_chunks(words, pos_tags):
        """Shuffle either the entire sentence or chunks within it."""
        n = len(words)
        transformed = words
        if n <= 3:
            return repeat_words(transformed, pos_tags)
            
        if (random.random() < 0.5) or (n <= 8):
            # Shuffle entire sentence
            random.shuffle(transformed)
        else:
            # Shuffle small segments
            chunk_size = random.randint(2, max(2, n // 3))
            start = random.randint(0, max(0, n - chunk_size))
            end = start + chunk_size
            chunk = transformed[start:end]
            random.shuffle(chunk)
            transformed = transformed[:start] + chunk + transformed[end:]
        return transformed

    transformed = words
    # Randomly decide which transformations to apply
    transformations = [repeat_words, swap_words, shuffle_chunks]
    num_to_apply = random.randint(1, 3)  # apply at least one transformation
    chosen = random.sample(transformations, num_to_apply)

    for func in chosen:
        transformed = func(transformed, pos_tags)

    return " ".join(transformed)


def retrieval_candidates(value):
    """Usable captions of a retrieval-negative field, without the "<empty>" marker."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or isinstance(value, str):
        return []
    return [c for c in value if isinstance(c, str) and c and c != EMPTY]


class BCEDataset(BEiTDataset):
    """Stage-1 pairs: the correct caption and one incorrect caption drawn from the word-level
    augmentation, the stored incorrect caption (`foil`) and the three retrieval bands.

    use_original_foil=True also offers the human-written incorrect caption of FOIL-it
    (`original_foil`, kept by the LLM step). The released checkpoints were trained
    without it.
    """
    def __init__(self, *args, use_original_foil=False, **kwargs):
        self.use_original_foil = use_original_foil
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
        data["image"] = (img,img)

        text_segment = item["caption"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)

        candidates = []
        # generated negative
        aug_negative = augment_negatives(
            words=item["tokenized_words"],
            pos_tags=item["pos_tagging"]
        )
        candidates.append(aug_negative)

        # stored incorrect caption (LLM-written, or the dataset's own)
        candidates.append(item["foil"])
        if self.use_original_foil and isinstance(item.get("original_foil"), str) \
                and item["original_foil"] != item["foil"]:
            candidates.append(item["original_foil"])

        # retrieval negatives, one per similarity band that has any
        for field in [
            "retrieval_hard_foil_list",
            "retrieval_medium_foil_list",
            "retrieval_easy_foil_list"
        ]:
            options = retrieval_candidates(item[field]) if field in item else []
            if options:
                candidates.append(random.choice(options))

        # ---------------------------------------------------------
        # 4. Randomly pick 1 foil from all candidates
        # ---------------------------------------------------------
        text_segment = random.choice(candidates)
        foil_language_tokens, foil_padding_mask, _ = self._get_text_segment(text_segment)

        data["language_tokens"] = (language_tokens, foil_language_tokens)
        data["padding_mask"] = (padding_mask, foil_padding_mask)
        data["targets"] = (torch.tensor([1], dtype=torch.float32),torch.tensor([0], dtype=torch.float32))
