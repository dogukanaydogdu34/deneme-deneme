# API

## Python

```python
from PIL import Image
import slidetext as st

st.CONFIG["inpaint"] = "auto"        # "auto" | "opencv" | "lama"

page = st.Page(Image.open("slayt.png").convert("RGB"))
page.boxes = st.detect(page.image)   # ya da st.detect(img, regions=[{"x":..,"y":..,"w":..,"h":..,"text":"..."}])

for b in page.boxes:
    print(b.id, b.text, b.font, b.weight, b.color, b.stroke_w, b.shadow)

edits = [
    {"id": 0, "text": "Yeni başlık"},                           # stil otomatik korunur
    {"id": 2, "text": "Bulut", "color": "#e0592f"},             # tek alan değiştir
    st.apply_preset(page, {"id": 1, "text": "Etiket"}, "etiket"),
    {"new": True, "x": 100, "y": 900, "h": 48, "text": "Ek yazı", "font": "Nunito", "weight": 700},
]
st.render(page, edits).save("yeni.png")
```

| Fonksiyon | Açıklama |
|---|---|
| `load_pages(bytes, filename) -> list[Image]` | PNG/JPG/WebP/PDF'i sayfa görsellerine çevirir |
| `detect(img, regions=None) -> list[TextBox]` | OCR ve stil analizi |
| `analyze_style(np_rgb, box)` | Tek kutunun stilini yerinde doldurur |
| `match_font(mask, text, candidates=None)` | `(font, kalınlık, italik, puan)` döndürür |
| `render(page, edits) -> Image` | Düzenlemeleri uygular |
| `erase(page, ids) -> Image` | Yalnızca silme (önbellekli) |
| `apply_preset(page, edit, key) -> dict` | Hazır stili düzenlemeye uygular |
| `preset_preview(key, color) -> Image` | 220×120 önizleme |
| `boxes_to_json / boxes_from_json` | Kutuları kaydetmek ve yüklemek için (OCR'ı tekrar çalıştırmadan) |

## Düzenleme (edit) nesnesi

Yalnızca değişen alanları göndermek yeterlidir. Gönderilmeyen alanlar algılanan stilden gelir.

| Alan | Tür | Açıklama |
|---|---|---|
| `id` | int | Kutu numarası (`detect` sırası) |
| `text` | str | Yeni metin. `""` yazıyı tamamen siler. `\n` satır atlatır. |
| `font`, `weight`, `italic` | str, int, bool | `FONTS` anahtarları. Kalınlık 100-1000, en yakın değere oturur. |
| `color` | `#rrggbb` | Dolgu rengi |
| `size` | int | px. 0 ya da yoksa orijinal boyuta oturur. |
| `align` | `left` / `center` / `right` | Hizalama |
| `dx`, `dy` | int | Konum kaydırma (px) |
| `fit` | bool | Taşarsa kutu genişliğine sığdırır |
| `line_height` | float | Varsayılan 1.25 |
| `stroke_w`, `stroke_color` | int, hex | Kontur |
| `stroke2_w`, `stroke2_color` | int, hex | Dış kontur (çıkartma) |
| `shadow`, `shadow_color`, `shadow_dx`, `shadow_dy` | bool, hex, int, int | Gölge |
| `glow`, `glow_color`, `glow_r` | bool, hex, int | Neon (`glow_r` 0 ise otomatik) |
| `extrude`, `extrude_color` | int, hex | 3B kabartma derinliği |
| `hollow` | bool | İçi boş harf |
| `box`, `box_color`, `box_pad` | bool, hex, int | Etiket arka planı |
| `rotate` | float | Derece |
| `new`, `x`, `y`, `w`, `h` | | Yeni yazı kutusu (silme yapılmaz) |

## TextBox (algılama çıktısı)

Düzenleme alanlarının hepsini içerir. Bunlara ek olarak şu alanlar vardır:

- `x, y, w, h`: dolgu mürekkebinin kutusu
- `ox, oy, ow, oh`: OCR'ın verdiği kutu
- `conf`: OCR güveni
- `font_score`: font benzerliği, 0-1 arası

## HTTP (server.py)

| Yöntem | Yol | Gövde / Yanıt |
|---|---|---|
| GET | `/api/fonts` | `{fonts:[{name,group,italic,weights}], presets:[{key,label}]}` |
| POST | `/api/upload` | multipart `file` → `{doc, name, pages:[{w,h}]}` |
| GET | `/api/{doc}/{n}/image` | Orijinal sayfa (PNG) |
| GET | `/api/{doc}/{n}/boxes` | `{boxes:[TextBox]}`. İlk çağrıda OCR çalışır. |
| POST | `/api/{doc}/{n}/render` | `{edits:[...]}` → PNG |
| POST | `/api/{doc}/{n}/preset` | `{key, edit}` → `{edit}` |
| GET | `/api/preset/{key}.png?color=rrggbb` | Önizleme |
| POST | `/api/{doc}/export` | `{format:"pdf"\|"png", edits:{"0":[...]}}` → dosya |

Belgeler bellekte tutulur. Sunucu yeniden başlarsa yükleme tekrarlanmalıdır.

## Komut satırı (cli.py)

```bash
python cli.py detect slayt.pdf -o kutular.json
python cli.py render slayt.pdf duzenlemeler.json --boxes kutular.json -o yeni.pdf
python cli.py replace slayt.png "Eski başlık=Yeni başlık" "Güneş=Bulut" --preset kitap -o yeni.png
```
