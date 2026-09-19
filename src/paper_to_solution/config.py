"""Central configuration. Secrets come from env vars only -- never hardcoded."""
from __future__ import annotations

import os

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "qwen/qwen3.8-27b")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_TIMEOUT_S", "60"))

# Image handling limits (conservative, prototype-safe)
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB (Groq rejects larger payloads)
MAX_IMAGE_DIM = 2048  # downscale longest side beyond this for latency/quality balance

# Vision call budget. Verified 2026-09-19 on the on_demand tier (OTPM 1000):
# any per-request budget above ~1000 output tokens is rejected outright, so
# the cap is 1000 and dense pages are handled by tile-splitting in the
# extractor (two overlapping halves, merged with page-boundary dedup).
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "1000"))
VISION_RENDER_DPI = int(os.environ.get("VISION_RENDER_DPI", "100"))

# One image per vision request (verified against installed SDK/API; batching = sequential calls)


def get_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or add it to .env "
            "(see .env.example). The key is never hardcoded."
        )
    return key
