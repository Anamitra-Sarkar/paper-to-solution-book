"""Shared test helper: render a synthetic question-paper image with PIL."""
from __future__ import annotations

from PIL import Image, ImageDraw


def make_paper_image(path, lines: list[str], width: int = 1200, line_h: int = 60):
    img = Image.new("RGB", (width, line_h * (len(lines) + 2)), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for line in lines:
        d.text((40, y), line, fill="black")
        y += line_h
    img.save(path)
    return path
