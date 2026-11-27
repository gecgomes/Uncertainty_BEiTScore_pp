"""FLEUR (Lee et al., 2024) and EXPERT (Kim et al., 2025): LLaVA-1.5-13B asked for a score from
0.0 to 1.0, smoothed with the probabilities of the digit tokens it generated (the score-smoothing
of FLEUR, also used by EXPERT). Prompts, models and decoding are those of the scripts used for
the BEiTScore paper.

Each metric runs on the LLaVA code its own repository pins (baselines/fleur/FLEUR/LLaVA, LLaVA
1.1.3, and baselines/expert/EXPERT/LLaVA, LLaVA 1.2.2), installed in its own environment by
baselines/fleur/setup.sh and baselines/expert/setup.sh.
The generated tokens are taken as the last len(scores) tokens of the output, which works whether
or not the LLaVA version returns the prompt as well. A reply without a number in [0, 1] gives NaN
(the original scripts skipped it, which shifted every later score by one).
"""
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def fleur_prompt(caption):
    return ("Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading "
            "Criteria. (Print Real Number Score ONLY)\n\nGrading Criteria:\n\n0.0: The caption does not describe "
            "the image at all.\n1.0: The caption accurately and clearly describes the image.\n\n"
            f"Caption: {caption}\n\nScore(Choose a rating from 0.0 to 1.0):")


def expert_prompt(caption):
    return f"Evaluate the caption and assign a score on a scale of 0.0 to 1.0.\n\nCaption: {caption}\n\nScore (0.0~1.0):"


class _LLaVAJudge:
    model_path = model_base = None
    prompt = None

    def __init__(self, opts):
        repo = os.path.abspath(opts.llava_dir)
        if not os.path.isdir(os.path.join(repo, "llava")):
            raise FileNotFoundError(f"LLaVA code not found in {repo}; run the setup.sh of this runner")
        sys.path.insert(0, repo)
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        self.c = SimpleNamespace(IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
                                 DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
                                 SeparatorStyle=SeparatorStyle, conv_templates=conv_templates,
                                 KeywordsStoppingCriteria=KeywordsStoppingCriteria, process_images=process_images,
                                 tokenizer_image_token=tokenizer_image_token)
        name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.model_path, self.model_base, name)
        self.rate2token = {s: self.tokenizer.encode(str(s))[-1] for s in range(10)}
        self.image_args = SimpleNamespace(image_aspect_ratio="pad")
        self.info = {"model_path": self.model_path, "model_base": self.model_base, "conv_mode": "llava_v1"}

    def _generate(self, image_path, caption):
        c = self.c
        conv = c.conv_templates["llava_v1"].copy()
        image = Image.open(image_path).convert("RGB")
        tensor = c.process_images([image], self.image_processor, self.image_args)
        tensor = [t.to(self.model.device, dtype=torch.float16) for t in tensor] if isinstance(tensor, list) \
            else tensor.to(self.model.device, dtype=torch.float16)
        prompt = self.prompt(caption)
        if self.model.config.mm_use_im_start_end:
            prompt = c.DEFAULT_IM_START_TOKEN + c.DEFAULT_IMAGE_TOKEN + c.DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt = c.DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = c.tokenizer_image_token(conv.get_prompt(), self.tokenizer, c.IMAGE_TOKEN_INDEX,
                                            return_tensors="pt").unsqueeze(0).to(self.model.device)
        stop = conv.sep if conv.sep_style != c.SeparatorStyle.TWO else conv.sep2
        criteria = c.KeywordsStoppingCriteria([stop], self.tokenizer, input_ids)
        with torch.inference_mode():
            out = self.model.generate(input_ids, images=tensor, do_sample=False, num_beams=1, max_new_tokens=512,
                                      use_cache=True, stopping_criteria=[criteria], output_scores=True,
                                      return_dict_in_generate=True)
        steps = out.scores
        new = out.sequences[0, -len(steps):]
        return new, steps

    def _digit_probs(self, steps, position):
        p = torch.softmax(steps[position].float(), dim=-1)[0]
        return {d: p[t].item() for d, t in self.rate2token.items()}

    def _positions(self, new, digit):
        return (new == self.rate2token[digit]).nonzero().flatten().tolist()

    def score_one(self, image_path, caption):
        new, steps = self._generate(image_path, caption)
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        number = re.sub(r"\.+$", "", re.sub(r"^\.+", "", re.sub(r"[^\d.]", "", text)))
        try:
            value = float(number)
        except ValueError:
            return float("nan")
        if not 0 <= value <= 1:
            return float("nan")
        if value == 1.0:
            pos = self._positions(new, 1)
            if not pos:
                return float("nan")
            p = self._digit_probs(steps, pos[0])
            return 0.9 * p[0] + p[1]
        digits = str(value).split(".")[1]
        d1 = int(digits[0])
        pos = self._positions(new, d1)
        if not pos:
            return float("nan")
        # "0.0x": the first "0" token is the integer part, the decimal is the second one
        pos1 = pos[1] if (d1 == 0 and len(pos) > 1) else pos[0]
        p = self._digit_probs(steps, pos1)
        score = sum(p[d] * d * 0.1 for d in range(10))
        if len(digits) > 1:
            d2 = int(digits[1])
            pos2 = [q for q in self._positions(new, d2) if q > pos1]
            if pos2:
                p2 = self._digit_probs(steps, pos2[0])
                score += sum(p2[d] * d * 0.01 for d in range(10))
        return score

    def score(self, images, captions):
        out = []
        for image, caption in zip(images, captions):
            try:
                out.append(self.score_one(image, caption))
            except Exception as e:
                print(f"  {type(self).__name__} failed on {image}: {e}")
                out.append(float("nan"))
        return np.asarray(out, dtype=float)


class FLEUR(_LLaVAJudge):
    model_path = "liuhaotian/llava-v1.5-13b"
    model_base = None
    prompt = staticmethod(fleur_prompt)


class EXPERT(_LLaVAJudge):
    model_path = "hjkim811/EXPERT-llava-13b-lora"
    model_base = "liuhaotian/llava-v1.5-13b"
    prompt = staticmethod(expert_prompt)
