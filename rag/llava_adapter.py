"""Lazy, CUDA-only LLaVA 1.5 adapter; never imports the Gradio application."""

import math

from .common import fingerprint
from .prompts import DESCRIPTION_PROMPT


class LlavaAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA requires the project's CUDA environment (torch and llava)"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no text-only or CPU fallback is used")
        try:
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "install LLaVA in the project's CUDA environment; see README"
            ) from exc
        self.torch = torch
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed_all(self.settings.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenize_image = tokenizer_image_token
        self.process_images = process_images
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.conv_template = conv_templates[self.settings.conversation]
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            self.settings.llava_model,
            None,
            get_model_name_from_path(self.settings.llava_model),
            device_map="cuda:0",
            device="cuda",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.context_len = min(
            context_len, getattr(self.model.config, "max_position_embeddings", context_len)
        )
        vision = self.model.get_vision_tower()
        self.image_tokens = int(vision.num_patches)
        if getattr(vision, "select_feature", "patch") == "cls_patch":
            self.image_tokens += 1
        if getattr(self.model.config, "image_aspect_ratio", None) == "anyres":
            raise ValueError("this adapter supports the fixed-image-token LLaVA 1.5 configuration")

    def signature(self):
        self.load()
        from importlib.metadata import PackageNotFoundError, version

        import llava
        import transformers

        try:
            llava_version = version("llava")
        except PackageNotFoundError:
            llava_version = getattr(llava, "__version__", "unknown")
        vision = self.model.get_vision_tower()
        return {
            "model_id": self.settings.llava_model,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "config": fingerprint(self.model.config.to_dict()),
            "processor": fingerprint(self.image_processor.to_dict()),
            "vocabulary": fingerprint(self.tokenizer.get_vocab()),
            "vision_config": fingerprint(vision.config.to_dict()),
            "context_length": self.context_len,
            "image_tokens": self.image_tokens,
            "attention_implementation": self.model.config._attn_implementation,
            "torch": self.torch.__version__,
            "transformers": transformers.__version__,
            "llava": llava_version,
        }

    def wrapped_prompt(self, prompt):
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )

        marker = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, "mm_use_im_start_end", False):
            marker = DEFAULT_IM_START_TOKEN + marker + DEFAULT_IM_END_TOKEN
        conversation = self.conv_template.copy()
        conversation.append_message(conversation.roles[0], marker + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    def prompt_ids(self, prompt):
        self.load()
        ids = self.tokenize_image(
            self.wrapped_prompt(prompt),
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        )
        if int((ids == self.image_token_index).sum()) != 1:
            raise ValueError("prompt must contain exactly one original query image marker")
        return ids

    def fits(self, prompt, reserve_tokens):
        ids = self.prompt_ids(prompt)
        # LLaVA replaces its single placeholder with the vision patch sequence.
        return len(ids) - 1 + self.image_tokens + reserve_tokens <= self.context_len

    def inputs(self, image, prompt, reserve_tokens):
        if not self.fits(prompt, reserve_tokens):
            raise ValueError("prompt plus image tokens exceeds LLaVA's context window")
        tensors = self.process_images([image], self.image_processor, self.model.config)
        if isinstance(tensors, list):
            tensors = [x.to(device="cuda:0", dtype=self.torch.float16) for x in tensors]
        else:
            tensors = tensors.to(device="cuda:0", dtype=self.torch.float16)
        return {
            "input_ids": self.prompt_ids(prompt).unsqueeze(0).to("cuda:0"),
            "images": tensors,
            "image_sizes": [image.size],
        }

    def generate(self, image, prompt, max_new_tokens):
        inputs = self.inputs(image, prompt, max_new_tokens)
        if max_new_tokens == 1:
            # One-step greedy generation needs no decoding cache. LLaVA's
            # generate override otherwise allocates KV for the entire prompt.
            with self.torch.inference_mode():
                output = self.model(**inputs, return_dict=True, use_cache=False)
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            return self.tokenizer.batch_decode(token, skip_special_tokens=True)[0].strip()
        # LLaVA's override names this positional argument `inputs`, not `input_ids`.
        prefix = inputs.pop("input_ids")
        with self.torch.inference_mode():
            output = self.model.generate(
                prefix,
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        # LLaVA versions differ: some return only generated IDs, others a prefix.
        if output.shape[1] >= prefix.shape[1] and self.torch.equal(
            output[:, : prefix.shape[1]], prefix
        ):
            output = output[:, prefix.shape[1] :]
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def describe(self, image):
        description = self.generate(image, DESCRIPTION_PROMPT, self.settings.description_max_tokens)
        description = description[: self.settings.description_max_chars].strip()
        if not description:
            raise ValueError("LLaVA generated an empty image description")
        return description

    def option_ids(self, choices):
        letters = sorted(choices)
        option_ids = []
        for letter in letters:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("option {} is not a single LLaVA token".format(letter))
            option_ids.append(ids[0])
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("supplied option letters do not have distinct token IDs")
        return letters, option_ids

    def draft_with_scores(self, image, prompt, choices):
        """Reuse one forward pass for the original greedy draft and APS logits."""
        inputs = self.inputs(image, prompt, 1)
        letters, option_ids = self.option_ids(choices)
        with self.torch.inference_mode():
            output = self.model(**inputs, return_dict=True, use_cache=False)
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            scores = output.logits[0, -1, option_ids].float().cpu().tolist()
        if not all(math.isfinite(x) for x in scores):
            raise ValueError("LLaVA returned non-finite draft option scores")
        raw = self.tokenizer.batch_decode(token, skip_special_tokens=True)[0].strip()
        return raw, dict(zip(letters, scores))

    def score_options(self, image, prompt, choices):
        inputs = self.inputs(image, prompt, 1)
        letters, option_ids = self.option_ids(choices)
        with self.torch.inference_mode():
            output = self.model(**inputs, return_dict=True, use_cache=False)
            scores = output.logits[0, -1, option_ids].float().cpu().tolist()
        if not all(math.isfinite(x) for x in scores):
            raise ValueError("LLaVA returned non-finite option scores")
        best = max(range(len(letters)), key=lambda i: scores[i])
        return letters[best], dict(zip(letters, scores))
