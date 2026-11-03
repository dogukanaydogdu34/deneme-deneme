"""Uçtan uca testler. Model indirmeden çalışır (OCR yerine bölge + metin verilir, doldurma OpenCV).

    pip install pytest && python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slidetext as st  # noqa: E402

st.CONFIG["inpaint"] = "opencv"


def _bg(W=1200, H=500):
    y, x = np.mgrid[0:H, 0:W]
    return Image.fromarray(np.stack([230 + 0 * x, 246 - y / H * 12, 246 - x / W * 10], -1).astype(np.uint8))


def _region(img, text, font, size, weight, **kw):
    """Metni çizer ve OCR kutusu gibi gevşek bir bölge döndürür."""
    d = ImageDraw.Draw(img)
    f = st.load_font(font, size, weight)
    x, y = 120, 220
    if kw.get("shadow"):
        dx, dy, c = kw["shadow"]
        d.text((x + dx, y + dy), text, font=f, fill=c, anchor="ls", stroke_width=kw.get("stroke", 0), stroke_fill=c)
    d.text((x, y), text, font=f, fill=kw.get("color", (60, 165, 175)), anchor="ls",
           stroke_width=kw.get("stroke", 0), stroke_fill=(255, 255, 255))
    l, t, r, b = f.getbbox(text, anchor="ls", stroke_width=kw.get("stroke", 0))
    m = 12
    return {"x": x + l - m, "y": y + t - m, "w": r - l + 2 * m, "h": b - t + 2 * m, "text": text}


def test_style_detection_outline_shadow():
    img = _bg()
    reg = _region(img, "Doğa ve Çevremiz", "Baloo 2", 110, 800, stroke=7, shadow=(5, 7, (30, 80, 90)))
    b = st.detect(img, [reg])[0]
    assert b.font == "Baloo 2" and b.weight >= 700
    assert 5 <= b.stroke_w <= 9
    assert b.shadow and abs(b.shadow_dx - 5) <= 2 and abs(b.shadow_dy - 7) <= 2


def test_font_match_plain():
    img = _bg()
    reg = _region(img, "Fotosentez Nasıl Çalışır", "Lora", 90, 700, color=(30, 40, 70))
    b = st.detect(img, [reg])[0]
    assert b.font == "Lora" and b.stroke_w == 0 and not b.shadow


def test_erase_leaves_no_trace():
    clean = _bg()
    img = clean.copy()
    reg = _region(img, "Silinecek yazı", "Inter", 80, 700, color=(200, 90, 40))
    page = st.Page(img)
    page.boxes = st.detect(img, [reg])
    out = np.asarray(st.render(page, [{"id": 0, "text": ""}]), np.float32)
    ref = np.asarray(clean, np.float32)
    x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
    assert np.abs(out[y:y + h, x:x + w] - ref[y:y + h, x:x + w]).mean() < 2.0


def test_replace_keeps_style():
    img = _bg()
    reg = _region(img, "Eski Başlık", "Baloo 2", 100, 800, stroke=6)
    page = st.Page(img)
    page.boxes = st.detect(img, [reg])
    out = np.asarray(st.render(page, [{"id": 0, "text": "Yeni Başlık"}]))
    # beyaz kontur pikselleri hâlâ var
    assert (out.reshape(-1, 3) > 250).all(1).sum() > 500


@pytest.mark.parametrize("key", list(st.PRESETS))
def test_presets(key):
    im = st.preset_preview(key)
    assert im.size == (220, 120)
