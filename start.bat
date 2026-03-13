@echo off
set ANONYMIZED_TELEMETRY=False

echo ──────────────────────────────
echo  RAG Locale — Avvio
echo ──────────────────────────────

curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠  Ollama non raggiungibile su localhost:11434
    echo    Avvialo con: ollama serve
    pause
    exit /b 1
)

echo ✓ Ollama attivo
echo   Avvio server su http://localhost:8000
echo.

python server.py
pause
