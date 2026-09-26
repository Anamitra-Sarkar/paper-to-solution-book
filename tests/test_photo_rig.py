"""Offline tests for the bad-photo fixture rig (no API calls)."""
from pathlib import Path

import pytest

ASSET = Path(__file__).parent / "assets" / "question_paper_455.pdf"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="testcase PDF not vendored")


@needs_asset
def test_degradation_is_deterministic(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import hashlib

    import make_bad_photos as mbp

    import fitz
    from PIL import Image
    import io as _io

    doc = fitz.open(ASSET)
    base = Image.open(_io.BytesIO(doc[1].get_pixmap(dpi=150).tobytes("png"))).convert("RGB")
    doc.close()
    fn = dict(mbp.VARIANTS)["blur_dim"]
    h1 = hashlib.sha256(fn(base.copy(), __import__("random").Random(3)).tobytes()).hexdigest()
    h2 = hashlib.sha256(fn(base.copy(), __import__("random").Random(3)).tobytes()).hexdigest()
    assert h1 == h2


@needs_asset
def test_ground_truth_matches_parser(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import make_bad_photos as mbp

    gt = mbp.ground_truth(ASSET, 2)
    assert gt["count"] == 5  # Q4-Q8 on page 2 of paper 455
    assert set(gt["marks"]) == {"4", "5", "6", "7", "8"}
    assert all(m == 1 for m in gt["marks"].values())


@needs_asset
def test_ten_variants_defined():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import make_bad_photos as mbp

    assert len(mbp.VARIANTS) == 10
