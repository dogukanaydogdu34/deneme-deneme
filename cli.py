"""Komut satırı: arayüz olmadan toplu / otomatik kullanım.

  # 1) Yazıları ve stillerini çıkar (sayfa başına kutu listesi)
  python cli.py detect slayt.pdf -o kutular.json

  # 2) Düzenlemeleri uygula (kutular.json verilirse OCR tekrar çalışmaz)
  python cli.py render slayt.pdf duzenlemeler.json -o yeni.pdf --boxes kutular.json

  # 3) Kısayol: metni metinle değiştir
  python cli.py replace slayt.png "Doğa ve Çevremiz=Canlılar ve Doğa" -o yeni.png --preset kitap

duzenlemeler.json biçimi: {"0": [{"id": 0, "text": "Yeni"}], "1": [...]}  (anahtar = sayfa no)
Tek sayfada liste de olur: [{"id": 0, "text": "Yeni"}]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slidetext as st  # noqa: E402


def _pages(path):
    with open(path, "rb") as fh:
        return [st.Page(im) for im in st.load_pages(fh.read(), path)]


def _save(images, out):
    ext = os.path.splitext(out)[1].lower()
    if ext == ".pdf":
        images[0].save(out, "PDF", save_all=True, append_images=images[1:], resolution=144)
    elif len(images) == 1:
        images[0].save(out)
    else:
        base, e = os.path.splitext(out)
        for i, im in enumerate(images, 1):
            im.save(f"{base}_{i:02d}{e}")
    print("kaydedildi:", out)


def _load_boxes(pages, boxes_path):
    if boxes_path:
        data = json.load(open(boxes_path, encoding="utf-8"))
        for i, p in enumerate(pages):
            p.boxes = st.boxes_from_json(data.get(str(i), []))
    else:
        for p in pages:
            p.boxes = st.detect(p.image)


def cmd_detect(a):
    pages = _pages(a.input)
    out = {}
    for i, p in enumerate(pages):
        p.boxes = st.detect(p.image)
        out[str(i)] = st.boxes_to_json(p.boxes)
        for b in p.boxes:
            print(f"[s{i} #{b.id}] {b.text!r}  font={b.font} {b.weight}  renk={b.color}"
                  f"{'  kontur=' + str(b.stroke_w) if b.stroke_w else ''}{'  gölge' if b.shadow else ''}")
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("kaydedildi:", a.output)


def cmd_render(a):
    pages = _pages(a.input)
    _load_boxes(pages, a.boxes)
    edits = json.load(open(a.edits, encoding="utf-8"))
    if isinstance(edits, list):
        edits = {"0": edits}
    _save([st.render(p, edits.get(str(i), [])) for i, p in enumerate(pages)], a.output)


def cmd_replace(a):
    pages = _pages(a.input)
    _load_boxes(pages, a.boxes)
    pairs = [r.split("=", 1) for r in a.pairs]
    images = []
    for p in pages:
        edits = []
        for old, new in pairs:
            texts = [b.text for b in p.boxes]
            hit = difflib.get_close_matches(old, texts, n=1, cutoff=0.6)
            if not hit:
                continue
            b = p.boxes[texts.index(hit[0])]
            e = {"id": b.id, "text": new}
            if a.preset:
                e = st.apply_preset(p, e, a.preset)
            edits.append(e)
            print(f"{hit[0]!r} -> {new!r}")
        images.append(st.render(p, edits))
    _save(images, a.output)


def main():
    ap = argparse.ArgumentParser(description="Slayt Metin Düzenleyici - komut satırı")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="yazıları ve stilleri JSON'a çıkar")
    d.add_argument("input")
    d.add_argument("-o", "--output", default="kutular.json")
    d.set_defaults(fn=cmd_detect)
    r = sub.add_parser("render", help="düzenleme JSON'unu uygula")
    r.add_argument("input")
    r.add_argument("edits")
    r.add_argument("-o", "--output", default="duzenlenmis.pdf")
    r.add_argument("--boxes", help="detect çıktısı (OCR'ı atlar, id'ler sabit kalır)")
    r.set_defaults(fn=cmd_render)
    p = sub.add_parser("replace", help='"eski=yeni" çiftleriyle hızlı değiştir')
    p.add_argument("input")
    p.add_argument("pairs", nargs="+")
    p.add_argument("-o", "--output", default="duzenlenmis.png")
    p.add_argument("--boxes")
    p.add_argument("--preset", choices=list(st.PRESETS), help="değişen yazılara hazır stil uygula")
    p.set_defaults(fn=cmd_replace)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
