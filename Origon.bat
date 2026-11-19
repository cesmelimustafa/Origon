@echo off
chcp 65001 >nul
title Origon - Kurumsal M^&A Full-Stack Web Platformu (Port 8000)

:: Bat dosyasinin bulundugu proje klasorune otomatik gec
cd /d "%~dp0"

echo ======================================================================
echo   ORIGON - Kurumsal Birlesme ve Satin Alma Full-Stack Web Platformu
echo   Backend : Python ASGI REST API (/api/v1/*)
echo   Frontend: Modern Web Terminali (web/index.html, CSS3, ES6 JS)
echo   Adres   : http://127.0.0.1:8000
echo ======================================================================
echo.

:: Eger egitilmis model artefaktlari yoksa once pipeline'i calistir
if not exist "artifacts\ranked_targets.parquet" (
    echo [1/3] Model artefaktlari bulunamadi, egitim pipeline'i baslatiliyor...
    python -m src.pipeline --force-rebuild
    if errorlevel 1 (
        echo.
        echo [HATA] Pipeline calistirilirken bir sorun olustu.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Hazir model ve SHAP artefaktlari bulundu.
)

:: Eski 8501 (Streamlit) ve 8000 (Origon API) portlarinda acik kalmis surecleri temizle
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/3] Tarayici 2 saniye icinde http://127.0.0.1:8000 adresinde acilacak...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

echo [3/3] Origon Full-Stack Web ve REST API Sunucusu (Port 8000) baslatiliyor...
echo.
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000

pause
