"""CLIP's shared image/text space, with explicit full-text chunking."""

import numpy as np

from .common import fingerprint


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if not np.all(np.isfinite(vector)) or np.any(norm <= 1e-12):
        raise ValueError("embedding must be finite and nonzero")
    return vector / norm


def fuse(image_vector, text_vector):
    image_vector, text_vector = unit(image_vector), unit(text_vector)
    if image_vector.ndim != 1 or image_vector.shape != text_vector.shape:
        raise ValueError("image and text embeddings must share one vector space")
    return unit(0.5 * image_vector + 0.5 * text_vector)


def token_chunks(tokenizer, text, max_length):
    """Tokenize without truncation, then reserve room for BOS/EOS per chunk."""
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("CLIP context is smaller than its special-token overhead")
    # The full sequence never goes to the model; suppress its misleading
    # unchunked-length warning while keeping truncation explicitly disabled.
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    parts = [ids[start : start + capacity] for start in range(0, len(ids), capacity)] or [[]]
    return [tokenizer.build_inputs_with_special_tokens(part) for part in parts]


def average_chunks(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("expected at least one CLIP chunk embedding")
    return unit(unit(vectors).mean(axis=0))


class ClipEncoder:
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP requires the project's CUDA torch/transformers environment"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no text-only or CPU fallback is used")
        self.torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.settings.clip_model)
        # Leave room for LLaVA's generation cache on a 16 GB GPU. CLIP moves to
        # CUDA only while embedding; this does not change its FP32 computation.
        self.model = CLIPModel.from_pretrained(self.settings.clip_model).float().eval()
        self.max_length = int(self.model.config.text_config.max_position_embeddings)

    def signature(self):
        self.load()
        import transformers

        return {
            "model_id": self.settings.clip_model,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "config": fingerprint(self.model.config.to_dict()),
            "processor": fingerprint(self.processor.image_processor.to_dict()),
            "vocabulary": fingerprint(self.processor.tokenizer.get_vocab()),
            "text_max_length": self.max_length,
            "torch": self.torch.__version__,
            "transformers": transformers.__version__,
        }

    def _image_vector(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with self.torch.inference_mode():
            vector = self.model.get_image_features(pixel_values=inputs["pixel_values"].to("cuda:0"))
        return unit(vector[0].float().cpu().numpy())

    def _text_vector(self, text):
        chunks = token_chunks(self.processor.tokenizer, text, self.max_length)
        vectors = []
        # Bound GPU memory even for unusually long hints; no chunk is dropped.
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            inputs = self.processor.tokenizer.pad(
                {"input_ids": batch, "attention_mask": [[1] * len(c) for c in batch]},
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                verbose=False,
            )
            with self.torch.inference_mode():
                encoded = self.model.get_text_features(
                    **{k: v.to("cuda:0") for k, v in inputs.items()}
                )
            vectors.extend(encoded.float().cpu().numpy())
        return average_chunks(vectors)

    def encode(self, image, text):
        self.load()
        self.torch.cuda.empty_cache()
        try:
            self.model.to("cuda:0")
            return fuse(self._image_vector(image), self._text_vector(text))
        finally:
            self.model.to("cpu")
            self.torch.cuda.empty_cache()
