"""Small, CPU-only types and artifact helpers."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

SCHEMA_VERSION = 1


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a JSON artifact only after its complete contents reach disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".rag-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def text(value):
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True)
class Query:
    """The complete query interface: deliberately no gold answer or logits."""

    question: str
    choices: Dict[str, str]
    hint: str = ""

    def __post_init__(self):
        if not text(self.question):
            raise ValueError("question must be nonempty")
        if len(self.choices) < 2:
            raise ValueError("at least two named choices are required")
        for label, value in self.choices.items():
            if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or not text(value):
                raise ValueError("choices must have single uppercase letters and nonempty text")


@dataclass(frozen=True)
class Settings:
    llava_model: str = "liuhaotian/llava-v1.5-7b"
    clip_model: str = "openai/clip-vit-base-patch32"
    conversation: str = "vicuna_v1"
    description_max_tokens: int = 64
    description_max_chars: int = 384
    draft_max_tokens: int = 1
    seed: int = 42

    def __post_init__(self):
        if min(self.description_max_tokens, self.description_max_chars, self.draft_max_tokens) <= 0:
            raise ValueError("generation bounds must be positive")

    def metadata(self):
        return {
            **asdict(self),
            "schema_version": SCHEMA_VERSION,
            "prompt_version": 1,
            "decoding": "greedy; num_beams=1",
            "draft_decoding": "one full-vocabulary greedy token; no KV cache",
            "llava_dtype": "float16",
            "llava_attention": "sdpa",
            "clip_dtype": "float32",
            "clip_residency": "CPU between calls; CUDA for all embeddings",
            "image_preprocessing": "PIL EXIF transpose, RGB; native model processor",
            "text_embedding": "all content tokens; mean of unit CLIP chunks; renormalize",
            "fusion": "unit(0.5 * unit(image) + 0.5 * unit(text))",
            "retrieval": "cosine; top 3; ascending numeric record ID breaks ties",
            "split": "union(index % 1000000, decoded RGB image hash); 80/20; seed 42",
        }
