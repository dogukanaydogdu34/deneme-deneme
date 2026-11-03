@echo off
chcp 65001 >nul
REM Slayt Metin Duzenleyici - Windows baslatici. Cift tikla.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Python yoneticisi ^(uv^) kuruluyor...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
if not exist .venv\Scripts\python.exe (
  echo Ilk kurulum: Python ve kutuphaneler indiriliyor, birkac dakika surer...
  uv venv --python 3.12 .venv
  uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  if errorlevel 1 ( echo Kurulum basarisiz & pause & exit /b 1 )
)
if not exist models\big-lama.pt .venv\Scripts\python.exe setup_models.py
.venv\Scripts\python.exe -W ignore server.py
pause
