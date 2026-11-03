"""Slayt metin düzenleme motoru: OCR -> stil analizi -> silme -> yeniden yazma.

Açık kaynak bileşenler:
  - EasyOCR (Apache-2.0)        : metin bulma ve okuma
  - OpenCV (Apache-2.0)         : renk kümeleme, kontur/gölge analizi, düz zeminde doldurma
  - LaMa big-lama (Apache-2.0)  : dokulu zeminde doldurma
  - Pillow + Google Fonts (OFL) : yeni metni basma, font eşleştirme
"""
from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass, asdict, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo kökü
FONT_DIR = os.environ.get("SLIDETEXT_FONT_DIR", os.path.join(BASE, "fonts"))
MODEL_PATH = os.environ.get("SLIDETEXT_LAMA_PATH", os.path.join(BASE, "models", "big-lama.pt"))

# --------------------------------------------------------------------------- entegrasyon ayarları
# Başka bir sistemle birleştirirken bu değerler dışarıdan değiştirilebilir:
#   import slidetext; slidetext.CONFIG["inpaint"] = "opencv"
CONFIG = {
    # "auto": zemine göre OpenCV ya da LaMa, "opencv": hep OpenCV (modelsiz), "lama": hep LaMa
    "inpaint": os.environ.get("SLIDETEXT_INPAINT", "auto"),
    # Düz zemin eşiği: halka piksellerinin degrade yüzeyinden sapması (std) bunun altındaysa OpenCV
    "smooth_threshold": 3.0,
    # Font aday kısıtlayıcı: f(crop_rgb: np.ndarray, text: str) -> list[str] | None
    # Örn. bir font sınıflandırma modeli (font-classify) en olası font adlarını döndürür,
    # şablon eşleştirme yalnızca bunlar arasından en iyi font + kalınlığı seçer.
    "font_candidates": None,
    # OCR yerine dış sistemden kutu verilecekse detect(..., boxes=[...]) kullan.
    "ocr_langs": ["tr", "en"],
}

# --------------------------------------------------------------------------- fontlar
# var    : değişken font dosyası, w = (min, max) kalınlık aralığı
# static : {kalınlık: dosya}
# italic : italik değişken dosya (varsa)
# group  : arayüzde gruplama
FONTS = {
    # yuvarlak / çocuk kitabı tarzı (NotebookLM'de sık)
    "Baloo 2":          dict(var="Baloo2[wght].ttf", w=(400, 800), group="Yuvarlak"),
    "Nunito":           dict(var="Nunito[wght].ttf", italic="Nunito-Italic[wght].ttf", w=(200, 1000), group="Yuvarlak"),
    "Quicksand":        dict(var="Quicksand[wght].ttf", w=(300, 700), group="Yuvarlak"),
    "Comfortaa":        dict(var="Comfortaa[wght].ttf", w=(300, 700), group="Yuvarlak"),
    "Varela Round":     dict(static={400: "VarelaRound-Regular.ttf"}, group="Yuvarlak"),
    "M PLUS Rounded":   dict(static={700: "MPLUSRounded1c-Bold.ttf"}, group="Yuvarlak"),
    "Rubik":            dict(var="Rubik[wght].ttf", w=(300, 900), group="Yuvarlak"),
    # sans
    "Inter":            dict(var="Inter[opsz,wght].ttf", italic="Inter-Italic[opsz,wght].ttf", w=(100, 900), group="Sans"),
    "DM Sans":          dict(var="DMSans[opsz,wght].ttf", w=(100, 1000), group="Sans"),
    "Montserrat":       dict(var="Montserrat[wght].ttf", w=(100, 900), group="Sans"),
    "Poppins":          dict(static={400: "Poppins-Regular.ttf", 700: "Poppins-Bold.ttf"}, group="Sans"),
    "Open Sans":        dict(var="OpenSans[wdth,wght].ttf", w=(300, 800), group="Sans"),
    "Oswald":           dict(var="Oswald[wght].ttf", w=(200, 700), group="Sans"),
    # serif
    "Lora":             dict(var="Lora[wght].ttf", italic="Lora-Italic[wght].ttf", w=(400, 700), group="Serif"),
    "Playfair Display": dict(var="PlayfairDisplay[wght].ttf", italic="PlayfairDisplay-Italic[wght].ttf", w=(400, 900), group="Serif"),
    "Merriweather":     dict(var="Merriweather[opsz,wdth,wght].ttf", w=(300, 900), group="Serif"),
    "Roboto Slab":      dict(var="RobotoSlab[wght].ttf", w=(100, 900), group="Serif"),
    # dekoratif
    "Luckiest Guy":     dict(static={400: "LuckiestGuy-Regular.ttf"}, group="Dekoratif"),
    "Bebas Neue":       dict(static={400: "BebasNeue-Regular.ttf"}, group="Dekoratif"),
    "Patrick Hand":     dict(static={400: "PatrickHand-Regular.ttf"}, group="Dekoratif"),
}
DEFAULT_FONT = "Inter"


def font_weights(name: str) -> list[int]:
    spec = FONTS[name]
    if "static" in spec:
        return sorted(spec["static"])
    lo, hi = spec["w"]
    return sorted({max(lo, min(hi, w)) for w in (400, 600, 700, 800)})


# --------------------------------------------------------------------------- modeller
_lock = threading.Lock()
_reader = None
_lama = None


def reader():
    global _reader
    with _lock:
        if _reader is None:
            import easyocr
            _reader = easyocr.Reader(CONFIG["ocr_langs"], gpu=_has_cuda(), verbose=False)
    return _reader


def lama():
    global _lama
    with _lock:
        if _lama is None:
            import torch
            dev = "cuda" if _has_cuda() else "cpu"
            _lama = torch.jit.load(MODEL_PATH, map_location=dev).eval()
    return _lama


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- veri
@dataclass
class TextBox:
    id: int
    x: int
    y: int
    w: int
    h: int
    text: str
    color: str = "#000000"
    font: str = DEFAULT_FONT
    weight: int = 400
    italic: bool = False
    size: int = 0              # px, render sırasında hesaplanır
    conf: float = 0.0
    # kontur
    stroke_w: int = 0
    stroke_color: str = "#ffffff"
    # gölge
    shadow: bool = False
    shadow_color: str = "#000000"
    shadow_dx: int = 0
    shadow_dy: int = 0
    font_score: float = 0.0    # font eşleşme güveni (0-1)
    align: str = "left"
    # ek efektler (hazır stillerden ya da elle)
    stroke2_w: int = 0         # dış kontur (çıkartma görünümü)
    stroke2_color: str = "#ffffff"
    glow: bool = False         # neon parlama
    glow_color: str = "#ffd84d"
    glow_r: int = 0
    extrude: int = 0           # 3B kabartma derinliği (px)
    extrude_color: str = "#000000"
    hollow: bool = False       # içi boş harf
    box: bool = False          # etiket arka planı
    box_color: str = "#ffffff"
    box_pad: int = 0
    rotate: float = 0.0        # derece
    # OCR'ın verdiği (gevşek) kutu; x,y,w,h ise dolgu mürekkebine sıkıştırılmış kutu
    ox: int = 0
    oy: int = 0
    ow: int = 0
    oh: int = 0


STYLE_KEYS = ("text", "font", "weight", "italic", "color", "stroke_w", "stroke_color",
              "shadow", "shadow_color", "shadow_dx", "shadow_dy", "align",
              "stroke2_w", "stroke2_color", "glow", "glow_color", "glow_r", "extrude", "extrude_color",
              "hollow", "box", "box_color", "box_pad", "rotate")


@dataclass
class Page:
    image: Image.Image
    boxes: list[TextBox] = field(default_factory=list)
    _erased: dict = field(default_factory=dict)  # frozenset(id) -> Image


# --------------------------------------------------------------------------- yükleme
def load_pages(data: bytes, filename: str) -> list[Image.Image]:
    if filename.lower().endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        for p in doc:
            zoom = max(1.0, 1920 / p.rect.width)
            pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return pages
    return [Image.open(io.BytesIO(data)).convert("RGB")]


# --------------------------------------------------------------------------- yardımcılar
def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in c)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def _expanded(arr, b: TextBox, frac=0.35):
    H, W = arr.shape[:2]
    g = max(4, int(b.oh * frac))
    x0, y0 = max(0, b.ox - g), max(0, b.oy - g)
    x1, y1 = min(W, b.ox + b.ow + g), min(H, b.oy + b.oh + g)
    inner = (b.ox - x0, b.oy - y0, b.ox - x0 + b.ow, b.oy - y0 + b.oh)
    return arr[y0:y1, x0:x1], (x0, y0), inner


# --------------------------------------------------------------------------- OCR
def detect(img: Image.Image, regions: list[dict] | None = None) -> list[TextBox]:
    """Sayfadaki yazıları bulur ve her birinin stilini analiz eder.

    regions verilmezse EasyOCR tüm sayfayı tarar.
    regions verilirse OCR algılaması atlanır; her bölge {"x","y","w","h", "text"?} olarak
    dışarıdan gelir (ör. kullanıcının çizdiği kutu ya da başka bir OCR'ın çıktısı).
    "text" yoksa bölge EasyOCR tanıma modeliyle okunur."""
    arr = np.array(img.convert("RGB"))
    raw = []
    if regions is None:
        res = reader().readtext(arr, paragraph=False, width_ths=0.7, text_threshold=0.6)
        for quad, text, conf in res:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            raw.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), text, float(conf)))
    else:
        for r in regions:
            x0, y0 = int(r["x"]), int(r["y"])
            x1, y1 = x0 + int(r["w"]), y0 + int(r["h"])
            text, conf = r.get("text"), float(r.get("conf", 1.0))
            if not text:
                rec = reader().recognize(arr[y0:y1, x0:x1])
                text, conf = (rec[0][1], float(rec[0][2])) if rec else ("", 0.0)
            raw.append((x0, y0, x1, y1, text, conf))
    boxes = []
    for x0, y0, x1, y1, text, conf in raw:
        if x1 - x0 < 4 or y1 - y0 < 4 or not str(text).strip() or (regions is None and conf < 0.2):
            continue
        b = TextBox(id=len(boxes), x=x0, y=y0, w=x1 - x0, h=y1 - y0, text=text, conf=conf,
                    ox=x0, oy=y0, ow=x1 - x0, oh=y1 - y0)
        try:
            analyze_style(arr, b)
        except Exception as e:  # stil analizi başarısız olsa da kutu kullanılabilir kalsın
            print("stil analizi atlandı:", text, e)
        boxes.append(b)
    W = arr.shape[1]
    for b in boxes:  # sayfanın ortasına oturan yazılar ortalı kabul edilir
        if abs(b.x + b.w / 2 - W / 2) < W * 0.03 and b.w < W * 0.9:
            b.align = "center"
    if regions is None:
        boxes.sort(key=lambda b: (round((b.y + b.h / 2) / 24), b.x))
    for i, b in enumerate(boxes):
        b.id = i
    return boxes


def boxes_to_json(boxes: list[TextBox]) -> list[dict]:
    return [asdict(b) for b in boxes]


def boxes_from_json(data: list[dict]) -> list[TextBox]:
    names = set(TextBox.__dataclass_fields__)
    return [TextBox(**{k: v for k, v in d.items() if k in names}) for d in data]


def _fill_pixels(crop: np.ndarray, inner) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harf DOLGUSU piksellerini bulur (kontur ve gölge hariç).
    k-means (k=4): kenarda en az, OCR kutusu içinde en çok görülen küme = dolgu rengi.
    Döner: (dolgu maskesi, dolgu rengi, arka plan rengi)."""
    import cv2
    h, w = crop.shape[:2]
    ix0, iy0, ix1, iy1 = inner
    px = crop.reshape(-1, 3).astype(np.float32)
    k = 4 if len(px) > 200 else 2
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.reshape(h, w)
    ring = np.zeros((h, w), bool)
    ring[:2] = ring[-2:] = True
    ring[:, :2] = ring[:, -2:] = True
    inside = np.zeros((h, w), bool)
    inside[iy0:iy1, ix0:ix1] = True
    bg_id = int(np.bincount(lab[ring], minlength=k).argmax())
    best, score = None, -1.0
    for c in range(k):
        if c == bg_id:
            continue
        fin = (lab[inside] == c).mean()
        fb = (lab[ring] == c).mean()
        if not (0.03 < fin < 0.8):
            continue
        # iç kısımda çok, kenarda az + harf gibi kalın gövde (dolgu) olmalı
        m = (lab == c) & inside
        body = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        thick = float(np.percentile(body[m], 90)) if m.any() else 0
        sc = fin / (fb + 0.02) * (1 + thick / max(1, (iy1 - iy0)) * 8)
        if sc > score:
            best, score = c, sc
    if best is None:
        best = (bg_id + 1) % k
    color = cen[best]
    bgc = cen[bg_id]
    d_t = np.linalg.norm(crop.astype(np.float32) - color, axis=2)
    d_b = np.linalg.norm(crop.astype(np.float32) - bgc, axis=2)
    fg = ((lab == best) | ((d_t < 0.35 * d_b) & (d_t < 70))) & inside
    # küçük gürültüleri at
    n, cc, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(4, (iy1 - iy0) ** 2 // 400)
    fg = keep[cc]
    return fg, color, bgc


def analyze_style(arr: np.ndarray, b: TextBox):
    """Dolgu rengi, kontur, gölge ve en yakın fontu tahmin eder."""
    import cv2
    crop, (cx, cy), inner = _expanded(arr, b)
    F, fill_c, _ = _fill_pixels(crop, inner)
    if F.sum() < 8:
        return
    cropf = crop.astype(np.float32)
    er = cv2.erode(F.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    b.color = _hex(np.median(cropf[er] if er.sum() > 20 else cropf[F], axis=0))
    ys, xs = np.where(F)
    fx0, fx1, fy0, fy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    b.x, b.y, b.w, b.h = cx + int(fx0), cy + int(fy0), int(fx1 - fx0), int(fy1 - fy0)

    # --- kontur: dolgunun etrafındaki halkalara dışa doğru bak
    dist_out = cv2.distanceTransform((~F).astype(np.uint8), cv2.DIST_L2, 5)
    R = max(3, b.h // 4)
    far = (dist_out > R + 2) & (dist_out <= R + 6)
    bg_c = np.median(cropf[far], axis=0) if far.sum() > 10 else np.median(cropf[~F], axis=0)
    fill_v = np.median(cropf[F], axis=0)
    rings, width = [], 0
    for r in range(1, R + 1):
        ring = (dist_out > r - 1) & (dist_out <= r)
        if ring.sum() < 10:
            break
        med = np.median(cropf[ring], axis=0)
        cons = float((np.linalg.norm(cropf[ring] - med, axis=1) < 40).mean())
        rings.append((r, med, cons))
    if len(rings) >= 2:
        ref = rings[1][1]  # 1. halka kenar yumuşatmasıdır; 2. halkayı referans al
        if _dist(ref, bg_c) > 18 and _dist(ref, fill_v) > 40 and rings[1][2] > 0.6:
            for r, med, cons in rings[1:]:
                if _dist(med, ref) < 0.6 * _dist(ref, bg_c) + 8 and cons > 0.5:
                    width = r
                else:
                    break
            if width >= 2:
                sel = (dist_out > 1) & (dist_out <= width)
                b.stroke_w = int(width)
                b.stroke_color = _hex(np.median(cropf[sel], axis=0))
    # --- gölge: (dolgu + kontur) maskesinin kaydırılmış kopyası arka plandan farklı mı?
    T = dist_out <= max(1, b.stroke_w)
    diff = np.linalg.norm(cropf - bg_c, axis=2) > 30
    s = max(2, b.h // 6)
    best, bscore = None, 0.0
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            if abs(dx) + abs(dy) < 2:
                continue
            sh = np.zeros_like(T)
            ys0, ys1 = max(0, dy), T.shape[0] + min(0, dy)
            xs0, xs1 = max(0, dx), T.shape[1] + min(0, dx)
            sh[ys0:ys1, xs0:xs1] = T[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            zone = sh & ~T
            if zone.sum() < max(20, T.sum() * 0.08):
                continue
            sc = float(diff[zone].mean())
            if sc > bscore:
                best, bscore = (dx, dy, zone), sc
    if best and bscore > 0.75:
        dx, dy, zone = best
        # karşı yönde de aynı şey varsa bu gölge değil kalın kontur/arka plandır
        opp = np.zeros_like(T)
        ys0, ys1 = max(0, -dy), T.shape[0] + min(0, -dy)
        xs0, xs1 = max(0, -dx), T.shape[1] + min(0, -dx)
        opp[ys0:ys1, xs0:xs1] = T[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        oz = opp & ~T
        if oz.sum() == 0 or float(diff[oz].mean()) < bscore - 0.35:
            b.shadow, b.shadow_dx, b.shadow_dy = True, int(dx), int(dy)
            b.shadow_color = _hex(np.median(cropf[zone & diff], axis=0))

    # --- konturlu/gölgeli yazılarda OCR yanılabilir: sadece dolguyu siyah-beyaz okut
    if (b.stroke_w or b.shadow or b.conf < 0.6) and b.conf < 0.995:  # dışarıdan metin verildiyse atla
        try:
            pad = max(6, b.h // 4)
            m = np.pad(~F[fy0:fy1, fx0:fx1], pad, constant_values=True)
            r = reader().recognize((m * 255).astype(np.uint8))
            if r and r[0][2] > b.conf and r[0][1].strip():
                b.text, b.conf = r[0][1].strip(), float(r[0][2])
        except Exception:
            pass
    # --- font eşleştirme
    cands = None
    if CONFIG.get("font_candidates"):
        try:
            cands = CONFIG["font_candidates"](crop[fy0:fy1, fx0:fx1], b.text)
        except Exception as e:
            print("font_candidates hatası:", e)
    b.font, b.weight, b.italic, b.font_score = match_font(F[fy0:fy1, fx0:fx1], b.text, cands)


# --------------------------------------------------------------------------- font eşleştirme
_render_cache: dict = {}


def _glyph_mask(name, weight, italic, text, H=48):
    key = (name, weight, italic, text, H)
    if key in _render_cache:
        return _render_cache[key]
    f = load_font(name, 72, weight, italic)
    l, t, r, bt = f.getbbox(text, anchor="ls")
    if r - l < 2 or bt - t < 2:
        return None
    im = Image.new("L", (r - l + 4, bt - t + 4), 0)
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, font=f, fill=255, anchor="ls")
    a = np.asarray(im) > 127
    ys, xs = np.where(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w = max(2, int(round(a.shape[1] * H / a.shape[0])))
    m = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize((w, H), Image.BILINEAR)) > 127
    if len(_render_cache) > 4000:
        _render_cache.clear()
    _render_cache[key] = m
    return m


def match_font(F: np.ndarray, text: str, candidates: list[str] | None = None):
    """Dolgu maskesini aday fontlarla çizilmiş aynı metinle karşılaştırır (şekil benzerliği)."""
    text = text.strip()
    if len(text) < 2 or F.shape[0] < 6:
        return DEFAULT_FONT, 400, False, 0.0
    H = 48
    w0 = max(2, int(round(F.shape[1] * H / F.shape[0])))
    tgt = np.asarray(Image.fromarray(F.astype(np.uint8) * 255).resize((w0, H), Image.BILINEAR)) > 127
    tgt_fill = tgt.mean()
    best = (DEFAULT_FONT, 400, False, -1.0)
    names = [n for n in (candidates or []) if n in FONTS] or list(FONTS)
    for name in names:
        spec = FONTS[name]
        for italic in ([False, True] if spec.get("italic") else [False]):
            for wt in font_weights(name):
                m = _glyph_mask(name, wt, italic, text, H)
                if m is None:
                    continue
                ar = np.log(m.shape[1] / w0)
                mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w0, H), Image.BILINEAR)) > 127
                inter = (mm & tgt).sum()
                iou = inter / max(1, (mm | tgt).sum())
                # en-boy oranı ve mürekkep yoğunluğu (kalınlık) farkı ceza
                score = iou * np.exp(-3.0 * abs(ar)) * np.exp(-4.0 * abs(mm.mean() - tgt_fill))
                if score > best[3]:
                    best = (name, wt, italic, float(score))
    return best


# --------------------------------------------------------------------------- silme
def text_mask(arr: np.ndarray, b: TextBox, others=()) -> tuple[np.ndarray, tuple[int, int]]:
    """Yazının dolgu + kontur + gölge piksellerini kapsayan (genişletilmiş) maske."""
    import cv2
    crop, (x0, y0), inner = _expanded(arr, b)
    F, _, _ = _fill_pixels(crop, inner)
    T = F
    if b.stroke_w:
        T = cv2.distanceTransform((~F).astype(np.uint8), cv2.DIST_L2, 5) <= b.stroke_w + 1
    if b.shadow:
        dx, dy = b.shadow_dx, b.shadow_dy
        sh = np.zeros_like(T)
        ys0, ys1 = max(0, dy), T.shape[0] + min(0, dy)
        xs0, xs1 = max(0, dx), T.shape[1] + min(0, dx)
        sh[ys0:ys1, xs0:xs1] = T[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        T = T | sh
    k = max(5, b.h // 5) | 1
    m = cv2.dilate(T.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    for o in others:
        if o.id == b.id:
            continue
        ax0, ay0 = max(0, o.x - 1 - o.stroke_w - x0), max(0, o.y - 1 - o.stroke_w - y0)
        ax1 = min(m.shape[1], o.x + o.w + 1 + o.stroke_w - x0)
        ay1 = min(m.shape[0], o.y + o.h + 1 + o.stroke_w - y0)
        if ax1 > ax0 and ay1 > ay0:
            m[ay0:ay1, ax0:ax1] = 0
    return m.astype(bool), (x0, y0)


def _fill(arr: np.ndarray, mask_full: np.ndarray, bbox) -> None:
    """Maske alanını doldurur. Düz/degrade zeminde OpenCV, dokulu zeminde LaMa."""
    import cv2
    H, W = arr.shape[:2]
    mx0, my0, mx1, my1 = bbox
    pad = max(32, (my1 - my0))
    x0, y0, x1, y1 = max(0, mx0 - pad), max(0, my0 - pad), min(W, mx1 + pad), min(H, my1 + pad)
    crop = arr[y0:y1, x0:x1]
    msk = mask_full[y0:y1, x0:x1].astype(np.uint8)
    ring = cv2.dilate(msk, np.ones((9, 9), np.uint8)) & (1 - msk)
    ring_px = crop[ring.astype(bool)].astype(np.float32)
    rough = 0.0
    if ring.any():
        ry, rx = np.where(ring.astype(bool))
        A = np.c_[rx, ry, rx * ry, rx ** 2, ry ** 2, np.ones_like(rx)].astype(np.float32)
        coef, *_ = np.linalg.lstsq(A, ring_px, rcond=None)
        rough = float((ring_px - A @ coef).std())
    mode = CONFIG.get("inpaint", "auto")
    if mode == "opencv" or (mode == "auto" and rough < CONFIG.get("smooth_threshold", 3.0)):
        out = cv2.inpaint(crop, msk * 255, 5, cv2.INPAINT_NS)
    else:
        out = _lama_fill(crop, msk)
        if len(ring_px):
            shift = ring_px.mean(0) - out[ring.astype(bool)].astype(np.float32).mean(0)
            out = np.clip(out.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(msk.astype(np.float32), (5, 5), 1.2)[..., None]
    arr[y0:y1, x0:x1] = (out * soft + crop * (1 - soft)).astype(np.uint8)


def _lama_fill(crop: np.ndarray, msk: np.ndarray) -> np.ndarray:
    import torch
    h, w = crop.shape[:2]
    scale = min(1.0, 768 / max(h, w))
    sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
    ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8
    c_img = np.asarray(Image.fromarray(crop).resize((sw, sh), Image.LANCZOS), np.float32) / 255.0
    c_msk = (np.asarray(Image.fromarray(msk * 255).resize((sw, sh), Image.BILINEAR)) > 20).astype(np.float32)
    t_img = np.pad(c_img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t_msk = np.pad(c_msk, ((0, ph), (0, pw)))
    m = lama()
    dev = next(m.parameters()).device
    ti = torch.from_numpy(t_img).permute(2, 0, 1)[None].to(dev)
    tm = torch.from_numpy(t_msk)[None, None].to(dev)
    with torch.no_grad():
        out = m(ti, tm)[0].permute(1, 2, 0).cpu().numpy()
    out = (np.clip(out[:sh, :sw], 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(out).resize((w, h), Image.LANCZOS))


def erase(page: Page, ids: set[int]) -> Image.Image:
    key = frozenset(ids)
    if key in page._erased:
        return page._erased[key]
    best = frozenset()
    for k in page._erased:
        if k <= key and len(k) > len(best):
            best = k
    arr = np.array(page._erased.get(best, page.image))
    orig = np.asarray(page.image)
    H, W = arr.shape[:2]
    byid = {b.id: b for b in page.boxes}
    for i in sorted(key - best):
        b = byid[i]
        m, (x0, y0) = text_mask(orig, b, [o for o in page.boxes if o.id not in key])
        full = np.zeros((H, W), bool)
        full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        if not full.any():
            continue
        ys, xs = np.where(full)
        _fill(arr, full, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    out = Image.fromarray(arr)
    page._erased[key] = out
    return out


# --------------------------------------------------------------------------- yazma
def load_font(name: str, size: float, weight: int = 400, italic: bool = False):
    return _load_font(name, max(4, int(size)), int(weight), bool(italic))


from functools import lru_cache


@lru_cache(maxsize=512)
def _font_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@lru_cache(maxsize=2048)
def _load_font(name: str, size: int, weight: int, italic: bool):
    spec = FONTS.get(name) or FONTS[DEFAULT_FONT]
    if "static" in spec:
        wt = min(spec["static"], key=lambda w: abs(w - weight))
        return ImageFont.truetype(io.BytesIO(_font_bytes(os.path.join(FONT_DIR, spec["static"][wt]))), size)
    path = spec["italic"] if (italic and spec.get("italic")) else spec["var"]
    f = ImageFont.truetype(io.BytesIO(_font_bytes(os.path.join(FONT_DIR, path))), size)
    try:
        vals = []
        for a in f.get_variation_axes():
            nm = a.get("name", b"")
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            if "eight" in nm:
                vals.append(max(a["minimum"], min(a["maximum"], weight)))
            elif "ptical" in nm:
                vals.append(max(a["minimum"], min(a["maximum"], size)))
            else:
                vals.append(a["default"])
        f.set_variation_by_axes(vals)
    except Exception:
        pass
    return f


def fit_size(b: TextBox, ref_text: str) -> int:
    """Orijinal metnin dolgu yüksekliği kutu yüksekliğine eşit olacak font boyutu."""
    lo, hi = 4, max(8, b.h * 3)
    for _ in range(14):
        mid = (lo + hi) / 2
        f = load_font(b.font, mid, b.weight, b.italic)
        l, t, r, bt = f.getbbox(ref_text, anchor="ls")
        if bt - t > b.h:
            hi = mid
        else:
            lo = mid
    return int(lo)


def render(page: Page, edits: list[dict]) -> Image.Image:
    """edits: [{id, text, <STYLE_KEYS>, size(0=otomatik), dx, dy, fit}]
    Yeni metin kutusu: {"new": true, "x", "y", "h", "text", ...} (silme yapılmaz)."""
    byid = {b.id: b for b in page.boxes}
    changed = [e for e in edits if not e.get("new") and e.get("id") in byid and _is_changed(byid[e["id"]], e)]
    added = [e for e in edits if e.get("new") and e.get("text", "").strip()]
    canvas = erase(page, {e["id"] for e in changed}).convert("RGBA")
    for e in changed:
        if e.get("text", "").strip():
            _draw(canvas, byid[e["id"]], e)
    for e in added:
        _draw(canvas, new_box(e), {**e, "size": e.get("size") or int(e.get("h", 40))})
    return canvas.convert("RGB")


def new_box(e: dict) -> TextBox:
    h = int(e.get("h", 40))
    return TextBox(id=-1, x=int(e["x"]), y=int(e["y"]), w=int(e.get("w", 400)), h=h, text="Hg",
                   color=e.get("color", "#000000"))


def _rgba(c, a=255):
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a)


def _draw(canvas: Image.Image, orig: TextBox, e: dict):
    from PIL import ImageFilter
    b = TextBox(**{**asdict(orig), **{k: e[k] for k in STYLE_KEYS if k in e and k != "text"}})
    auto = fit_size(b, orig.text)
    size = int(e.get("size") or 0) or auto
    lines = e["text"].split("\n")
    f = load_font(b.font, size, b.weight, b.italic)
    if e.get("fit", False) and not e.get("size"):
        widest = max(lines, key=lambda t: f.getlength(t))
        l, _, r, _ = f.getbbox(widest, anchor="ls")
        if r - l > orig.w * 1.02:
            size = max(4, int(size * orig.w / (r - l)))
            f = load_font(b.font, size, b.weight, b.italic)
    k = size / max(1, auto) if not orig.id == -1 else 1.0  # boyut değişince efektler de ölçeklensin
    S = lambda v: int(round(float(v) * k))
    sw, sw2 = S(b.stroke_w), S(b.stroke2_w)
    _, t0, _, _ = f.getbbox(orig.text, anchor="ls")
    baseline = orig.y - t0 + int(e.get("dy", 0))
    lh = float(e.get("line_height", 1.25)) * size + 2 * (sw + sw2)
    pos = []
    for i, line in enumerate(lines):
        if not line:
            continue
        l, t, r, bt = f.getbbox(line, anchor="ls")
        tw = r - l
        if b.align == "center":
            x = orig.x + (orig.w - tw) / 2 - l
        elif b.align == "right":
            x = orig.x + orig.w - tw - l
        else:
            x = orig.x - l
        pos.append((x + int(e.get("dx", 0)), baseline + i * lh, line))
    if not pos:
        return

    W, H = canvas.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def text_layer(fill, stroke=0, stroke_fill=None, dx=0, dy=0):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for x, y, line in pos:
            d.text((x + dx, y + dy), line, font=f, fill=fill, anchor="ls",
                   stroke_width=stroke, stroke_fill=stroke_fill)
        return im

    outer = sw + sw2
    # 1) etiket arka planı
    if b.box:
        xs0 = min(x + f.getbbox(t, anchor="ls")[0] for x, y, t in pos)
        xs1 = max(x + f.getbbox(t, anchor="ls")[2] for x, y, t in pos)
        ys0 = min(y + f.getbbox(t, anchor="ls")[1] for x, y, t in pos)
        ys1 = max(y + f.getbbox(t, anchor="ls")[3] for x, y, t in pos)
        p = S(b.box_pad) or int(size * 0.28)
        ImageDraw.Draw(layer).rounded_rectangle(
            (xs0 - p - outer, ys0 - p * 0.8 - outer, xs1 + p + outer, ys1 + p * 0.8 + outer),
            radius=int(size * 0.22), fill=_rgba(b.box_color))
    # 2) neon parlama
    if b.glow:
        r = S(b.glow_r) or max(4, size // 5)
        a = text_layer(_rgba(b.glow_color), outer + r // 3, _rgba(b.glow_color)).getchannel("A")
        a = a.filter(ImageFilter.GaussianBlur(r))
        a = Image.fromarray(np.clip(np.asarray(a, np.float32) * 1.6, 0, 255).astype(np.uint8))
        g = Image.new("RGBA", (W, H), _rgba(b.glow_color))
        g.putalpha(a)
        layer = Image.alpha_composite(layer, g)
    # 3) 3B kabartma ya da gölge
    if b.extrude:
        depth = S(b.extrude)
        for i in range(depth, 0, -1):
            layer = Image.alpha_composite(layer, text_layer(_rgba(b.extrude_color), outer, _rgba(b.extrude_color), i, i))
    if b.shadow:
        sc = _rgba(b.shadow_color)
        layer = Image.alpha_composite(layer, text_layer(sc, outer, sc, S(b.shadow_dx), S(b.shadow_dy)))
    # 4) dış kontur (çıkartma)
    if sw2:
        layer = Image.alpha_composite(layer, text_layer(_rgba(b.stroke2_color), outer, _rgba(b.stroke2_color)))
    # 5) ana yazı
    if b.hollow:
        ring = text_layer(_rgba(b.color), max(2, sw or size // 18), _rgba(b.color))
        hole = text_layer((255, 255, 255, 255)).getchannel("A")
        a = np.asarray(ring.getchannel("A"), np.int16) - np.asarray(hole, np.int16)
        ring.putalpha(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
        main = ring
    else:
        main = text_layer(_rgba(b.color), sw, _rgba(b.stroke_color) if sw else None)
    layer = Image.alpha_composite(layer, main)
    # 6) döndürme (yazının merkezi etrafında)
    if b.rotate:
        bb = layer.getbbox()
        if bb:
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            layer = layer.rotate(float(b.rotate), resample=Image.BICUBIC, center=(cx, cy))
    canvas.alpha_composite(layer)


def _is_changed(b: TextBox, e: dict) -> bool:
    for k in STYLE_KEYS:
        if k in e and e[k] != getattr(b, k):
            return True
    return bool(e.get("size") or e.get("dx") or e.get("dy"))


# --------------------------------------------------------------------------- hazır stiller
# Her stil, yazının mevcut boyutuna (s) ve rengine (c) göre bir değişiklik seti üretir.
def _darker(c, f=0.55):
    r, g, bl, _ = _rgba(c)
    return _hex((r * f, g * f, bl * f))


def _lighter(c, f=0.6):
    r, g, bl, _ = _rgba(c)
    return _hex((r + (255 - r) * f, g + (255 - g) * f, bl + (255 - bl) * f))


_RESET = dict(stroke_w=0, stroke2_w=0, shadow=False, shadow_dx=0, shadow_dy=0, glow=False, glow_r=0,
              extrude=0, hollow=False, box=False, box_pad=0, rotate=0.0)

PRESETS = {
    "sade":      ("Sade",            lambda s, c: dict()),
    "kontur":    ("Beyaz kontur",    lambda s, c: dict(stroke_w=max(2, s // 14), stroke_color="#ffffff")),
    "golge":     ("Yumuşak gölge",   lambda s, c: dict(shadow=True, shadow_color=_darker(c, 0.35),
                                                       shadow_dx=max(2, s // 22), shadow_dy=max(2, s // 16))),
    "kitap":     ("Çocuk kitabı",    lambda s, c: dict(stroke_w=max(2, s // 16), stroke_color="#ffffff", shadow=True,
                                                       shadow_color=_darker(c, 0.45), shadow_dx=max(2, s // 25),
                                                       shadow_dy=max(3, s // 18))),
    "cikartma":  ("Çıkartma",        lambda s, c: dict(stroke_w=max(2, s // 16), stroke_color="#ffffff",
                                                       stroke2_w=max(3, s // 10), stroke2_color=c,
                                                       color=_lighter(c, 0.15))),
    "neon":      ("Neon",            lambda s, c: dict(glow=True, glow_color=c, glow_r=max(4, s // 6),
                                                       color=_lighter(c, 0.55))),
    "bos":       ("İçi boş",         lambda s, c: dict(hollow=True, stroke_w=max(2, s // 20))),
    "bos_golge": ("İçi boş + gölge", lambda s, c: dict(hollow=True, stroke_w=max(2, s // 20), shadow=True,
                                                       shadow_color=_lighter(c, 0.55), shadow_dx=max(3, s // 14),
                                                       shadow_dy=max(3, s // 14))),
    "kabartma":  ("3B kabartma",     lambda s, c: dict(extrude=max(3, s // 10), extrude_color=_darker(c, 0.5))),
    "etiket":    ("Etiket",          lambda s, c: dict(box=True, box_color=_lighter(c, 0.75), rotate=-4.0,
                                                       color=_darker(c, 0.9))),
    "etiket_k":  ("Koyu etiket",     lambda s, c: dict(box=True, box_color=c, color="#ffffff")),
    "pop":       ("Pop",             lambda s, c: dict(stroke_w=max(2, s // 14), stroke_color=_darker(c, 0.35),
                                                       extrude=max(3, s // 12), extrude_color=_darker(c, 0.35),
                                                       color=_lighter(c, 0.1))),
}


def apply_preset(page: Page | None, edit: dict, key: str) -> dict:
    """Seçili yazıya hazır stili uygular; boyut otomatikse mevcut boyuttan hesaplar."""
    _, fn = PRESETS[key]
    if edit.get("new"):
        orig = new_box(edit)
        size = int(edit.get("size") or edit.get("h", 40))
    else:
        orig = {b.id: b for b in page.boxes}[edit["id"]]
        b = TextBox(**{**asdict(orig), **{k: edit[k] for k in STYLE_KEYS if k in edit and k != "text"}})
        size = int(edit.get("size") or 0) or fit_size(b, orig.text)
    color = edit.get("color", orig.color)
    out = {**edit, **_RESET, **fn(size, color)}
    if not edit.get("new") and not edit.get("size"):
        # efekt değerleri otomatik boyuta göre; boyut ölçeğini 1 tut
        pass
    return out


def preset_preview(key: str, color: str = "#3fa7b3", text: str = "Abc") -> Image.Image:
    """Arayüzdeki küçük önizleme kutucukları."""
    W, H, s = 220, 120, 64
    im = Image.new("RGBA", (W, H), (246, 245, 242, 255))
    e = {"new": True, "x": 0, "y": 0, "w": W, "h": s, "text": text, "font": "Baloo 2", "weight": 800,
         "color": color, "size": s, "align": "center"}
    e = {**e, **_RESET, **PRESETS[key][1](s, color)}
    f = load_font("Baloo 2", s, 800)
    _, t, _, bt = f.getbbox(text, anchor="ls")
    e["y"] = int((H - (bt - t)) / 2) - 4
    _draw(im, new_box(e), e)
    return im.convert("RGB")
