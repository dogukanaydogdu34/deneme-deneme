"""Slayt Metin Düzenleyici - yerel web sunucusu.

Çalıştırma:  python server.py   ->  http://localhost:7860
"""
from __future__ import annotations

import io
import os
import uuid
import zipfile
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

import slidetext.engine as engine

app = FastAPI(title="Slayt Metin Düzenleyici")
DOCS: dict[str, dict] = {}  # doc_id -> {"name": str, "pages": [engine.Page]}
STATIC = os.path.join(engine.BASE, "static")


def _png(img) -> Response:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def _page(doc: str, n: int) -> engine.Page:
    d = DOCS.get(doc)
    if not d or not (0 <= n < len(d["pages"])):
        raise HTTPException(404, "Sayfa bulunamadı")
    return d["pages"][n]


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/fonts")
def fonts():
    return {"fonts": [{"name": k, "group": v.get("group", ""), "italic": bool(v.get("italic")),
                       "weights": engine.font_weights(k) if "static" in v else list(range(v["w"][0], v["w"][1] + 1, 100))}
                      for k, v in engine.FONTS.items()],
            "presets": [{"key": k, "label": v[0]} for k, v in engine.PRESETS.items()]}


_preview_cache: dict = {}


@app.get("/api/preset/{key}.png")
def preset_png(key: str, color: str = "3fa7b3"):
    if key not in engine.PRESETS:
        raise HTTPException(404)
    ck = (key, color)
    if ck not in _preview_cache:
        _preview_cache[ck] = engine.preset_preview(key, "#" + color.strip("#")[:6])
    r = _png(_preview_cache[ck])
    r.headers["Cache-Control"] = "max-age=3600"
    return r


class PresetReq(BaseModel):
    key: str
    edit: dict


@app.post("/api/{doc}/{n}/preset")
def apply_preset(doc: str, n: int, req: PresetReq):
    p = _page(doc, n)
    if req.key not in engine.PRESETS:
        raise HTTPException(404)
    return {"edit": engine.apply_preset(p, req.edit, req.key)}


@app.get("/fonts/{name}")
def font_file(name: str):
    path = os.path.join(engine.FONT_DIR, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path, media_type="font/ttf")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        imgs = engine.load_pages(data, file.filename or "slide.png")
    except Exception as e:  # bozuk dosya vb.
        raise HTTPException(400, f"Dosya açılamadı: {e}")
    doc = uuid.uuid4().hex[:10]
    DOCS[doc] = {"name": os.path.splitext(file.filename or "slayt")[0],
                 "pages": [engine.Page(im) for im in imgs]}
    return {"doc": doc, "name": DOCS[doc]["name"],
            "pages": [{"w": im.width, "h": im.height} for im in imgs]}


@app.get("/api/{doc}/{n}/image")
def page_image(doc: str, n: int):
    return _png(_page(doc, n).image)


@app.get("/api/{doc}/{n}/boxes")
def page_boxes(doc: str, n: int):
    p = _page(doc, n)
    if not p.boxes:
        p.boxes = engine.detect(p.image)
    return {"boxes": [asdict(b) for b in p.boxes]}


class Edits(BaseModel):
    edits: list[dict] = []


@app.post("/api/{doc}/{n}/render")
def page_render(doc: str, n: int, body: Edits):
    p = _page(doc, n)
    if not p.boxes:
        p.boxes = engine.detect(p.image)
    return _png(engine.render(p, body.edits))


class ExportReq(BaseModel):
    format: str = "pdf"            # "pdf" | "png"
    edits: dict[str, list[dict]] = {}  # sayfa no (str) -> edits


@app.post("/api/{doc}/export")
def export(doc: str, req: ExportReq):
    d = DOCS.get(doc)
    if not d:
        raise HTTPException(404)
    imgs = []
    for i, p in enumerate(d["pages"]):
        e = req.edits.get(str(i), [])
        if e and not p.boxes:
            p.boxes = engine.detect(p.image)
        imgs.append(engine.render(p, e) if e else p.image)
    name = d["name"] + "_duzenlenmis"
    if req.format == "pdf":
        buf = io.BytesIO()
        imgs[0].save(buf, "PDF", save_all=True, append_images=imgs[1:], resolution=144)
        return Response(buf.getvalue(), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})
    if len(imgs) == 1:
        buf = io.BytesIO()
        imgs[0].save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="{name}.png"'})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, im in enumerate(imgs, 1):
            b = io.BytesIO()
            im.save(b, "PNG")
            z.writestr(f"{name}_{i:02d}.png", b.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


if __name__ == "__main__":
    import uvicorn
    print("Modeller yükleniyor (ilk açılışta biraz sürer)...")
    engine.reader()
    engine.lama()
    port = int(os.environ.get("PORT", 7860))
    print(f"Hazır: http://localhost:{port}  (kapatmak için bu pencereyi kapat)")
    if not os.environ.get("NO_BROWSER"):
        import threading, webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
