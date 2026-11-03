"""slidetext: slayt görsellerindeki yazıları stilini koruyarak değiştirme.

Hızlı kullanım:
    from PIL import Image
    import slidetext as st

    page = st.Page(Image.open("slayt.png"))
    page.boxes = st.detect(page.image)                 # OCR + stil analizi
    out = st.render(page, [{"id": 0, "text": "Yeni başlık"}])
    out.save("yeni.png")
"""
from .engine import (  # noqa: F401
    CONFIG, FONTS, PRESETS, Page, TextBox, STYLE_KEYS,
    load_pages, detect, analyze_style, match_font, erase, render,
    apply_preset, preset_preview, boxes_to_json, boxes_from_json,
    load_font, font_weights, reader, lama,
)

__version__ = "0.3.0"
