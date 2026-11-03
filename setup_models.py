"""İlk kurulumda bir kez çalıştır: LaMa modelini (~200 MB) ve EasyOCR modellerini indirir."""
import os, urllib.request
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
dst = os.path.join(BASE, "models", "big-lama.pt")
if not os.path.exists(dst):
    url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
    print("LaMa modeli indiriliyor:", url)
    urllib.request.urlretrieve(url, dst)
print("EasyOCR modelleri hazırlanıyor (tr + en)...")
import easyocr
easyocr.Reader(["tr", "en"], gpu=False, verbose=False)
print("Tamam. Şimdi: python server.py")
