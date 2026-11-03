# Slayt Metin Düzenleyici (slidetext)

NotebookLM'den (ya da başka bir yerden) gelen slayt görsellerindeki yazıları değiştirmek için yerel araç.
Slaytı yüklersin, yazılar otomatik bulunur. Birine tıklayıp yeni metni yazarsın, araç eski yazıyı
arka planla doldurup yenisini basar. Sonucu PDF ya da PNG olarak indirirsin.

Her şey kendi bilgisayarında çalışır, hiçbir dosya dışarı gönderilmez.

## Kurulum ve çalıştırma

Bilgisayarda Python olması gerekmez. Başlatıcı gerekeni kendisi indirir.

- **Mac:** Klasördeki `Baslat (Mac).command` dosyasına sağ tıkla ve **Aç**'ı seç. Bunu yalnızca ilk seferde yapman gerekiyor, Mac'in güvenlik uyarısı için.
- **Windows:** `Baslat (Windows).bat` dosyasına çift tıkla. "Windows bilgisayarınızı korudu" uyarısı çıkarsa **Ek bilgi** ve ardından **Yine de çalıştır**'a tıkla.

İlk açılışta Python, kütüphaneler ve modeller indirilir. Bu adım internet hızına göre 5-15 dakika sürer ve yaklaşık 2 GB yer kaplar. Sonraki açılışlar bir dakikadan kısa sürer.
Hazır olunca tarayıcı kendiliğinden **http://localhost:7860** adresinde açılır. Kapatmak için siyah terminal penceresini kapat.

<details><summary>Elle kurulum (Python biliyorsan)</summary>

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py
python server.py
```
</details>

## Kullanım

1. **Slayt yükle**: NotebookLM'den indirdiğin PDF'i ya da PNG/JPG'yi sürükle. Çok sayfalı PDF'lerde sayfalar solda listelenir.
2. Slayttaki bir yazıya (ya da sağdaki listeden birine) tıkla, yeni metni yaz. Önizleme kendiliğinden güncellenir.
3. Font, kalınlık, italik, renk, boyut, hizalama ve konumu sağ panelden ayarla. Renk ve boyut otomatik tahmin edilir.
4. **Yazıyı sil**: yazıyı tamamen kaldırır. **+ Metin ekle**: slayta yeni yazı koyar.
5. **Hazır stiller**: tek tıkla Beyaz kontur, Yumuşak gölge, Çocuk kitabı, Çıkartma, Neon, İçi boş, 3B kabartma, Etiket, Pop. Renk, seçili yazının renginden türetilir.
6. **Efektler** bölümünde kontur, dış kontur, gölge, neon, kabartma, etiket, içi boş ve döndürme ayarları tek tek açılıp kapatılabilir.
7. **Orijinal** düğmesine basılı tutunca önceki hali görünür.
8. **PDF indir** tüm sayfaları birleştirir. **PNG indir** tek sayfada PNG, çok sayfada ZIP verir.

Satır atlamak için metin kutusunda Enter'a bas.

### Stil kopyalama

Yazıyı değiştirdiğinde araç orijinalin stilini korumaya çalışır:

- **Font:** yazının şekli 20 aday fontla karşılaştırılır, en benzer font ve kalınlık seçilir. Panelde benzerlik yüzdesi görünür.
- **Renk:** dolgu rengi.
- **Kontur:** varsa rengi ve kalınlığı.
- **Gölge:** varsa rengi ve kayma yönü.
- **Hizalama:** sayfanın ortasındaki yazılar ortalı kabul edilir.

Boyutu değiştirirsen kontur ve gölge de oranlı büyür ya da küçülür.

## Geliştiriciler için

```
slidetext/engine.py   motor: OCR, stil analizi, silme, yazma, hazır stiller
slidetext/__init__.py dışa açık Python API'si
server.py             FastAPI sunucusu + static/index.html arayüzü
cli.py                komut satırı (detect / render / replace)
tests/                model indirmeden çalışan testler (python -m pytest -q)
docs/ARCHITECTURE.md  akışın adım adım anlatımı
docs/API.md           Python, HTTP ve CLI API'si, edit şeması
docs/INTEGRATION.md   başka bir akışla (PaddleOCR / font-classify / canvas) birleştirme notları
fonts/                Google Fonts (OFL / Apache), lisanslar fonts/LICENSES.md içinde
models/               big-lama.pt (setup_models.py indirir, repoda yok)
```

Hızlı deneme:

```python
from PIL import Image
import slidetext as st
page = st.Page(Image.open("samples/kids_slide.png").convert("RGB"))
page.boxes = st.detect(page.image)
st.render(page, [{"id": 0, "text": "Canlılar ve Doğa"}]).save("cikti.png")
```

## Bileşenler ve lisanslar

| Adım | Bileşen | Lisans |
|---|---|---|
| Yazıyı bulma ve okuma | [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Türkçe + İngilizce) | Apache-2.0 |
| Renk kümeleme, kontur/gölge analizi, düz zeminde doldurma | OpenCV | Apache-2.0 |
| Dokulu zeminde doldurma | [LaMa](https://github.com/advimman/lama) (big-lama) | Apache-2.0 (kod); ağırlıklar Places2 ile eğitildi, ticari kullanım belirsiz |
| Yazı basma, font eşleştirme | Pillow + Google Fonts | MIT-CMU / OFL |
| PDF okuma | PyMuPDF | AGPL-3.0 (iç kullanımda sorun yok; dışarı açılacaksa pypdfium2 ile değiştirin) |
| Sunucu ve arayüz | FastAPI, Uvicorn, tek sayfa HTML | MIT / BSD |

## Font ekleme (ör. Madlen fontları)

1. `.ttf` / `.otf` dosyasını `fonts/` klasörüne koy.
2. `engine.py` içindeki `FONTS` sözlüğüne bir satır ekle:
   ```python
   # değişken font (dosya adında [wght] var):
   "Madlen Serif": dict(var="MadlenSerif[wght].ttf", italic=None, w=(400, 800), group="Madlen"),
   # sabit kalınlıklı dosyalar:
   "Madlen Sans": dict(static={400: "MadlenSans-Regular.ttf", 700: "MadlenSans-Bold.ttf"}, group="Madlen"),
   ```
3. Sunucuyu yeniden başlat. Font, otomatik eşleştirmeye de dahil olur.

## Bilinen sınırlar

- **Font tanıma listedeki fontlarla sınırlı.** Orijinal font listede yoksa en yakın olanı seçilir. Daha iyi eşleşme için `fonts/` klasörüne font ekleyebilirsin.
- **Algılanan efektler:** dolgu, kontur ve gölge. Neon, degrade ya da tek kelimede iki renk gibi stiller otomatik algılanmaz, hazır stillerden elle verilir.
- **Yazı tek satır tek kutudur.** Çok satırlı paragraflarda her satır ayrı kutu olarak gelir.
- **Hız.** CPU'da ilk OCR sayfa başına 15-30 sn sürer. Düzenlemeler düz zeminde yarım saniyenin altında, dokulu zeminde birkaç saniyede biter. NVIDIA GPU varsa otomatik kullanılır.
- Yazının üstüne bindiği karmaşık görsellerde (fotoğraf, ikon) doldurma kusurlu olabilir.
