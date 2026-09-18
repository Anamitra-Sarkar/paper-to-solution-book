"""Image loading / validation / lightweight preprocessing.

Reusable by direct uploads AND Yugal's PDF pipeline (rendered pages come here).
Philosophy: normalize without destroying quality -- resize only when oversized,
gentle contrast/autocontrast, no aggressive binarization.
"""
from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from paper_to_solution import config
from paper_to_solution.config import SUPPORTED_EXTENSIONS


class ImageError(ValueError):
    pass


def detect_mime(path: str | Path, raw: bytes | None = None) -> str:
    ext = Path(str(path)).suffix.lower() if path else ""
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if raw:
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if raw[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def load_image_bytes(path: str | Path) -> tuple[bytes, str]:
    """Load + validate an image file. Returns (bytes, mime). Raises ImageError."""
    p = Path(path)
    if not p.exists():
        raise ImageError(f"Image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageError(
            f"Unsupported image type '{p.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    raw = p.read_bytes()
    if not raw:
        raise ImageError(f"Image is empty: {path}")
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise ImageError(
            f"Image too large ({len(raw)/1e6:.1f} MB > {config.MAX_IMAGE_BYTES/1e6:.0f} MB): {path}"
        )
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except Exception as e:
        raise ImageError(f"Invalid/corrupt image {path}: {e}") from e
    return raw, detect_mime(p, raw)


def preprocess_image(raw: bytes) -> bytes:
    """Lightweight normalization: EXIF orientation, downscale-if-huge, gentle contrast.

    Never upscales, never binarizes. Returns PNG bytes.
    """
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > config.MAX_IMAGE_DIM:
            scale = config.MAX_IMAGE_DIM / longest
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # Gentle contrast lift helps phone photos of papers; harmless on clean scans.
        im = ImageEnhance.Contrast(im).enhance(1.15)
        im = ImageOps.autocontrast(im, cutoff=0.5)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def prepare_image_for_model(path: str | Path) -> tuple[str, dict]:
    """Full prep: load -> validate -> preprocess -> data URL. Returns (data_url, meta)."""
    raw, mime = load_image_bytes(path)
    processed = preprocess_image(raw)
    with Image.open(io.BytesIO(processed)) as im:
        size = im.size
    return to_data_url(processed), {
        "mime": mime,
        "original_bytes": len(raw),
        "processed_bytes": len(processed),
        "processed_size": size,
    }
