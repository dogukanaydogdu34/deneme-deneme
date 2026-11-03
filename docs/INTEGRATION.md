# Başka bir akışla birleştirme

Bu not, tarayıcıda çalışan şu akışla birleştirme için yazıldı:
PaddleOCR PP-OCRv6 (ONNX), LaMa ve MI-GAN (ONNX), storia/font-classify ve canvas üzerinde çizim.

## Adım adım karşılaştırma

| Adım | Diğer akış | Bu repo | Birleştirmede öneri |
|---|---|---|---|
| Yükleme | Tarayıcıda blob | Yerel sunucu, PDF ve çok sayfa | Tarayıcı tarafı kalsın. PDF için pdf.js ile sayfaları görsele çevirin. |
| Metin bulma | Kullanıcı kutu çiziyor, PaddleOCR okuyor | EasyOCR ile tüm sayfa otomatik | PaddleOCR'ın **algılama** modelini tüm sayfada çalıştırın. Çıkan kutular `detect(img, regions=...)` biçimine birebir uyar. |
| Stil analizi | Yok | Dolgu, kontur, gölge, renk, hizalama | **Bu repodan alınacak ana parça.** Aşağıda JS'e taşıma notları var. |
| Font | font-classify (ML) | Şablon eşleştirme (20 font) | **İkisini birleştirin:** sınıflandırıcı ilk 5-10 adayı versin, şablon eşleştirme bunlar arasından font ve kalınlığı seçsin. Hazır kanca: `CONFIG["font_candidates"]`. |
| Silme | LaMa, MI-GAN ya da OpenCV (elle seçim) | OpenCV NS ya da LaMa, zemine göre otomatik | Otomatik seçim mantığını (`_fill` içindeki zemin testi) alın. MI-GAN'ı da "orta" seçenek olarak tutun. |
| Maske | Kutu ya da fırça | Harf pikselleri + kontur + gölge, komşu yazılar korunur | `text_mask` mantığını alın. Kutu maskesine göre çok daha az iz bırakır. |
| Yazma | Canvas `fillText` | Pillow RGBA katmanları ve efektler | Canvas'ta karşılıkları aşağıda. |
| Hazır stiller | Yok | 12 stil | `PRESETS` saf veri ve formül, JS'e doğrudan taşınabilir. |
| Dışa aktarma | PNG, JPEG, WebP | PDF, PNG, ZIP | İkisini birleştirin. |

## Kanca 1: dış font sınıflandırıcı

```python
import slidetext as st

def font_candidates(crop_rgb, text):
    # crop_rgb: yazının kırpılmış görüntüsü (numpy, RGB)
    # font-classify modelinin ilk N tahmini -> FONTS'taki adlara eşle
    preds = my_font_classifier(crop_rgb)               # ["Baloo 2", "Nunito", ...]
    return [p for p in preds if p in st.FONTS]         # boş liste dönerse tüm fontlar denenir

st.CONFIG["font_candidates"] = font_candidates
```

Sınıflandırıcının tanıdığı fontların `.ttf` dosyalarını `fonts/` klasörüne koyup `FONTS` sözlüğüne eklemek gerekir. Şablon eşleştirme, karşılaştırmak için fontu çizebilmek zorunda.

## Kanca 2: dışarıdan kutu (PaddleOCR ya da kullanıcı çizimi)

```python
boxes = st.detect(img, regions=[{"x": 120, "y": 80, "w": 900, "h": 140, "text": "Doğa ve Çevremiz"}])
```

`text` verilmezse bölge EasyOCR tanıma modeliyle okunur. Verilirse OCR hiç çalışmaz.

## Kanca 3: doldurma yöntemi

```python
st.CONFIG["inpaint"] = "opencv"        # modelsiz, en hızlı
st.CONFIG["smooth_threshold"] = 3.0    # düz zemin eşiği
```

MI-GAN eklemek için `_fill()` içinde `_lama_fill` yanına aynı imzalı bir `_migan_fill(crop, msk)` yazıp `CONFIG["inpaint"] == "migan"` dalı eklemek yeterli.

## Stil analizini tarayıcıya taşımak (OpenCV.js)

| Python (bu repo) | Tarayıcı karşılığı |
|---|---|
| `cv2.kmeans` | `cv.kmeans` (OpenCV.js) ya da 20 satırlık basit bir JS k-means |
| `cv2.distanceTransform` | `cv.distanceTransform` |
| `cv2.connectedComponentsWithStats` | `cv.connectedComponentsWithStats` |
| `cv2.inpaint(..., INPAINT_NS)` | `cv.inpaint(src, mask, dst, 5, cv.INPAINT_NS)` |
| Kontur halkası taraması | `distanceTransform` çıktısında `r-1 < d <= r` olan piksellerin rengi |
| Gölge kaydırma testi | Maske dizisini (dx, dy) kaydırıp `ImageData` üzerinde karşılaştırma |

## Efektlerin canvas karşılıkları

| Efekt | Canvas 2D |
|---|---|
| Kontur | `ctx.lineWidth = 2*stroke_w; ctx.lineJoin = "round"; ctx.strokeText(); ctx.fillText()`. Pillow'daki `stroke_width` her yöne `stroke_w` kadar taşar. |
| Dış kontur | Önce `lineWidth = 2*(stroke_w + stroke2_w)` ile `strokeText`, sonra normal kontur |
| Gölge | Aynı metni `(dx, dy)` kaydırılmış olarak gölge rengiyle, konturuyla birlikte önce çiz |
| Neon | `ctx.shadowColor = glow; ctx.shadowBlur = r` ya da `ctx.filter = "blur(r px)"` ile ayrı katman |
| 3B kabartma | `for i = depth..1`: metni `(i, i)` kaydırarak kabartma rengiyle çiz |
| İçi boş | Yalnızca `strokeText`. Ya da `globalCompositeOperation = "destination-out"` ile dolguyu sil. |
| Etiket | `ctx.roundRect()` + `fill()`, metnin `measureText` kutusu ve iç boşlukla |
| Döndürme | `ctx.translate(cx, cy); ctx.rotate(deg * Math.PI / 180)` |
| Taban çizgisi | `ctx.textBaseline = "alphabetic"`. Konum hesabı için `_draw()` içindeki `baseline = orig.y - t0` mantığını kullanın. |

## Önerilen birleşik akış

1. **Tarayıcı:** Görsel ya da PDF yüklenir (pdf.js).
2. **PaddleOCR algılama** tüm sayfada çalışır. İstenirse kullanıcı kutu ekleyip silebilir.
3. **PaddleOCR tanıma** her kutuyu okur.
4. **Stil analizi** (bu repodaki mantık, OpenCV.js'e taşınmış hâli) her kutu için çalışır: dolgu maskesi, renk, kontur, gölge.
5. **Font:** font-classify ilk N adayı verir, şablon eşleştirme font ve kalınlığı seçer.
6. Kullanıcı metni ya da hazır stili değiştirir.
7. **Silme:** Maske = dolgu + kontur + gölge, genişletilmiş. Zemin düzse OpenCV NS, değilse MI-GAN ya da LaMa kullanılır.
8. Yazı canvas'ta efekt katmanlarıyla çizilir.
9. **Dışa aktarma:** PNG, JPEG, WebP ya da çok sayfalı PDF.

Tarayıcıya taşımak istemediğiniz parçalar için `server.py` HTTP API'si (`docs/API.md`) olduğu gibi kullanılabilir.
