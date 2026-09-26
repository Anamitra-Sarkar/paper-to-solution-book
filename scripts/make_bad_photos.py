"""Generate deterministic bad-phone-photo variants from real paper pages.

Takes rendered PDF pages and applies phone-camera degradations with a fixed
seed so the set is reproducible. Writes PNGs plus a manifest carrying ground
truth (question count + per-question marks from the digital text layer).

These are synthetic degradations of real papers, labelled as such in the
manifest -- a stand-in until real Set A phone photos are available.

Usage:
    python scripts/make_bad_photos.py --pdf tests/assets/question_paper_455.pdf \
        --pages 2 6 --out test_runs/bad_phones --seed 3
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402


def _shadow(img: Image.Image, rng: random.Random, side: str = "left") -> Image.Image:
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    px = overlay.load()
    for x in range(w):
        f = 1.0 - (x / w if side == "left" else (w - x) / w)
        v = int(110 * max(0.0, f) ** 1.5 * rng.uniform(0.85, 1.0))
        for y in range(h):
            px[x, y] = v
    black = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(
        Image.blend(img, black, 0.0), img,
        overlay.point(lambda v: 255 - v))


def _glare(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    cx, cy = int(w * rng.uniform(0.2, 0.8)), int(h * rng.uniform(0.2, 0.5))
    radius = int(min(w, h) * rng.uniform(0.15, 0.3))
    overlay = Image.new("L", (w, h), 0)
    px = overlay.load()
    for y in range(max(0, cy - radius), min(h, cy + radius)):
        for x in range(max(0, cx - radius), min(w, cx + radius)):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / radius
            if d < 1:
                px[x, y] = int(140 * (1 - d))
    white = Image.new("RGB", (w, h), (255, 255, 255))
    return Image.composite(white, img, overlay)


def _vignette_light(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    px = overlay.load()
    top = rng.uniform(40, 90)
    for y in range(h):
        v = int(top * max(0.0, 1.0 - y / (h * 0.55)))
        for x in range(w):
            px[x, y] = v
    black = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(img, black, overlay.point(lambda v: v))


def _jpeg_bytes(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


VARIANTS = [
    ("rot_blur", lambda img, rng: ImageEnhance.Brightness(
        img.rotate(3, expand=True, fillcolor="white").filter(
            ImageFilter.GaussianBlur(1.0))).enhance(0.95)),
    ("rot_jpeg", lambda img, rng: _jpeg_bytes(
        img.rotate(-2.5, expand=True, fillcolor="white"), 30)),
    ("blur_dim", lambda img, rng: ImageEnhance.Brightness(
        img.filter(ImageFilter.GaussianBlur(1.5))).enhance(0.85)),
    ("shadow", lambda img, rng: ImageEnhance.Contrast(
        _shadow(img, rng)).enhance(0.9)),
    ("glare_small", lambda img, rng: _jpeg_bytes(
        _glare(img, rng).resize(
            (int(img.size[0] * 0.7), int(img.size[1] * 0.7))), 60)),
    ("shear_blur", lambda img, rng: img.transform(
        img.size, Image.AFFINE, (1, 0.03, 0, 0.02, 1, 0),
        fillcolor="white").filter(ImageFilter.GaussianBlur(1.0))),
    ("jpeg_harsh", lambda img, rng: ImageEnhance.Contrast(
        _jpeg_bytes(img, 20)).enhance(1.2)),
    ("small_blur_jpeg", lambda img, rng: _jpeg_bytes(
        img.resize((img.size[0] // 2, img.size[1] // 2)).filter(
            ImageFilter.GaussianBlur(1.0)), 35)),
    ("toplight_rot", lambda img, rng: _vignette_light(
        img.rotate(1.5, expand=True, fillcolor="white"), rng)),
    ("crop_blur_jpeg", lambda img, rng: _jpeg_bytes(
        img.crop((int(img.size[0] * 0.05), int(img.size[1] * 0.05),
                  int(img.size[0] * 0.95), int(img.size[1] * 0.95))).filter(
            ImageFilter.GaussianBlur(2.0)), 25)),
]


def ground_truth(pdf_path: Path, page_no: int) -> dict:
    """Exact count + marks from the digital text layer (not from vision)."""
    import fitz

    from paper_to_solution.text_parser import parse_text_pages

    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    qs = parse_text_pages([(page_no, page.get_text())])
    doc.close()
    return {"count": len(qs),
            "marks": {q.question_number: q.marks for q in qs}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build bad-phone-photo fixtures.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="subset of variant names (default: all)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    import fitz

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pdf = Path(args.pdf)
    doc = fitz.open(pdf)
    manifest = {"synthetic": True,
                "note": "Deterministic degradations of real paper pages; "
                        "stand-in until real Set A phone photos land.",
                "seed": args.seed, "photos": []}
    variants = [v for v in VARIANTS if not args.variants or v[0] in args.variants]
    for page_no in args.pages:
        pix = doc[page_no - 1].get_pixmap(dpi=args.dpi)
        base = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        gt = ground_truth(pdf, page_no)
        for name, fn in variants:
            photo = fn(base.copy(), rng)
            fname = f"{pdf.stem}_p{page_no}_{name}.png"
            photo.save(out / fname)
            manifest["photos"].append(
                {"file": fname, "paper": pdf.stem, "page": page_no,
                 "variant": name, "expected_count": gt["count"],
                 "expected_marks": gt["marks"]})
            print(f"wrote {fname} ({photo.size[0]}x{photo.size[1]})")
    doc.close()
    manifest_path = out / "manifest.json"
    previous = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text()).get("photos", [])
        except Exception:
            previous = []
    seen = {p["file"] for p in manifest["photos"]}
    manifest["photos"].extend(p for p in previous if p["file"] not in seen)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest['photos'])} photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
