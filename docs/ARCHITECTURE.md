# Mimari ve akış

```
 yükleme ──► OCR ──► stil analizi ──► (kullanıcı düzenler) ──► silme ──► yeniden yazma ──► dışa aktarma
 PDF/PNG    EasyOCR   k-means + mesafe                         OpenCV NS   Pillow RGBA      PDF / PNG / ZIP
                      dönüşümü + font                          ya da LaMa  katmanları
                      şablon eşleştirme
```

Kodun tamamı `slidetext/engine.py` içinde. `server.py` bu motoru HTTP ile, `cli.py` komut satırıyla açar.
`static/index.html` tek dosyalık arayüzdür.

## 1. Yükleme — `load_pages()`

- PNG, JPG ve WebP olduğu gibi açılır.
- PDF sayfaları PyMuPDF ile yaklaşık 1920 px genişlikte görsele çevrilir. NotebookLM PDF'lerinde slaytlar zaten görsel olarak gelir.
- Her şey yerelde çalışır. Sunucu `127.0.0.1` adresini dinler ve dışarıya hiçbir istek atmaz.

## 2. Metin bulma — `detect(img, regions=None)`

- **Varsayılan:** EasyOCR (CRAFT algılama ve Latin tanıma, `tr` ve `en`) tüm sayfayı tarar. Güveni 0.2'nin altındaki sonuçlar elenir.
- **`regions` verilirse:** algılama atlanır. Kutular dışarıdan gelir, örneğin kullanıcının çizdiği kutular ya da başka bir OCR'ın çıktısı. Metin verilmemişse bölge yalnızca tanıma modeliyle okunur.
- **Yeniden okuma:** Konturlu ya da gölgeli yazılarda OCR sık yanılır. Bu yazılarda dolgu maskesi siyah-beyaz bir görüntüye çevrilip tekrar okutulur. Güven artarsa yeni sonuç kullanılır.

## 3. Stil analizi — `analyze_style()`

Her kutu, kutu yüksekliğinin %35'i kadar genişletilip kırpılır.

| Özellik | Yöntem |
|---|---|
| Dolgu pikselleri | k-means, k=4. Kenarda en çok görülen küme arka plan sayılır. İçeride çok ve kenarda az görülen, kalın gövdeli küme dolgudur (`_fill_pixels`). Küçük bileşenler atılır. |
| Dolgu rengi | Aşındırılmış dolgu piksellerinin medyanı. Kenar yumuşatması rengi etkilemez. |
| Kontur | Dolgunun dışına mesafe dönüşümü uygulanır ve 1 px'lik halkalar taranır. 2. halka arka plandan ve dolgudan farklıysa kontur vardır. Halka rengi tutarlı kaldığı sürece kalınlık artırılır. |
| Gölge | (dolgu + kontur) maskesi ±h/6 aralığında kaydırılır. Kaydırılmış bölgenin arka plandan farklı olma oranı en yüksek yön aranır. Ters yönde de aynı şey varsa bu gölge değildir. |
| Font ve kalınlık | `match_font`: OCR metni 20 fontun her kalınlığıyla 48 px yüksekliğe çizilir. Puan = IoU × en-boy cezası × mürekkep yoğunluğu cezası. |
| Hizalama | Yazının merkezi sayfa merkezinin ±%3'ü içindeyse ortalı kabul edilir. |

Kutunun `x, y, w, h` değerleri dolgu mürekkebine sıkıştırılır. OCR'ın gevşek kutusu `ox, oy, ow, oh` alanlarında saklanır.

## 4. Silme — `erase()` / `_fill()`

1. **Maske:** dolgu, kontur (`stroke_w + 1`) ve kaydırılmış gölge birleştirilir, sonra h/5 kadar genişletilir. Komşu yazıların mürekkep kutuları maskeden çıkarılır, böylece yan satır kırpılmaz.
2. **Zemin testi:** maskenin çevresindeki 9 px'lik halkaya 2. derece bir yüzey (degrade) oturtulur. Kalan sapmanın standart sapması 3'ün altındaysa zemin düzdür.
   - **Düz ya da degrade zeminde:** `cv2.inpaint(..., INPAINT_NS)`. Modelsizdir, milisaniyeler sürer ve iz bırakmaz.
   - **Dokulu zeminde:** LaMa (big-lama TorchScript). Kırpıntı en fazla 768 px'e küçültülür. Sonuçtaki renk kayması halka ortalamasıyla düzeltilir.
3. **Harmanlama:** Sonuç, yumuşatılmış maskeyle orijinale karıştırılır.
4. **Önbellek:** Silinmiş hâller `frozenset(id)` anahtarıyla saklanır. Yeni bir yazı silinirken en büyük alt küme önbellekten alınır, böylece düzenlemeler hızlı yenilenir.

## 5. Yeniden yazma — `_draw()`

Yazı tam sayfa boyutunda RGBA katmanlarla, arkadan öne doğru çizilir:

1. etiket kutusu (`box`)
2. neon parlaması (`glow`): yalnızca alfa kanalı Gauss bulanıklığından geçer, koyu hale oluşmaz
3. 3B kabartma (`extrude`) ve gölge (`shadow`)
4. dış kontur (`stroke2_w`)
5. ana yazı: dolgu ve kontur. İçi boş (`hollow`) seçiliyse konturdan dolgu alfası çıkarılır.
6. döndürme (`rotate`), yazının merkezi etrafında

**Boyut:** Otomatik modda, orijinal metnin dolgu yüksekliği kutuya eşit olacak şekilde ikili arama yapılır (`fit_size`). Kullanıcı boyutu değiştirirse kontur, gölge ve kabartma aynı oranla ölçeklenir.

**Taban çizgisi:** Orijinal metnin mürekkep üstü kutunun üstüne hizalanır. Bu sayede yeni metinde ğ ya da Ç gibi harfler olsa da satır kaymaz.

## 6. Hazır stiller — `PRESETS`, `apply_preset()`

Her stil `(boyut, renk) -> değişiklik sözlüğü` biçiminde bir fonksiyondur. Kontur kalınlığı ya da gölge mesafesi gibi değerler yazı boyutuna göre hesaplanır. Renkler seçili yazının renginden türetilir (`_darker`, `_lighter`). Yeni bir stil eklemek için `PRESETS` sözlüğüne tek satır yazmak yeterlidir.

## 7. Dışa aktarma

`server.py /api/{doc}/export` çok sayfalı PDF ya da PNG verir (çok sayfada ZIP). `cli.py` dosya uzantısına göre kaydeder.

## Performans (CPU, 2 çekirdek)

| Adım | Süre |
|---|---|
| İlk açılışta model yükleme | ~10 sn |
| OCR + stil analizi | sayfa başına 15-25 sn |
| Düz zeminde düzenleme | < 0.5 sn |
| Dokulu zeminde düzenleme (LaMa) | 3-5 sn |

NVIDIA GPU varsa otomatik olarak kullanılır.
