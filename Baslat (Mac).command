#!/bin/bash
# Slayt Metin Düzenleyici - Mac başlatıcı. Çift tıkla.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Python yöneticisi (uv) kuruluyor..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "uv kurulamadı"; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x .venv/bin/python ]; then
  echo "İlk kurulum: Python ve kütüphaneler indiriliyor (birkaç dakika sürer)..."
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt || { echo "Kurulum başarısız"; read -r; exit 1; }
fi
[ -f models/big-lama.pt ] || .venv/bin/python setup_models.py || { echo "Model indirilemedi"; read -r; exit 1; }
.venv/bin/python -W ignore server.py
