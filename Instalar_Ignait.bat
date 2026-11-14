@echo off
chcp 65001 >nul
title Ignait Studio v1.0 - Instalador Oficial [By CRESKO // Jhon Córdova]
color 0B

echo ============================================================================
echo    IGNAIT STUDIO v1.0 - INSTALADOR OFICIAL PARA WINDOWS
echo    Autor: Jhon Córdova Martínez (Alias: Cresko · VE)
echo    GitHub: https://github.com/MCJ-A/Ignait
echo ============================================================================
echo.

cd /d "%~dp0"

if exist "Ignait.exe" if exist "installer\install_ignait.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install_ignait.ps1"
    exit /b %ERRORLEVEL%
)

if exist "dist\Ignait\Ignait.exe" if exist "installer\install_ignait.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install_ignait.ps1"
    exit /b %ERRORLEVEL%
)

echo [Modo Desarrollador] No se detecto Ignait.exe compilado; instalando entorno Python...
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual Python (.venv)...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)

echo [2/3] Instalando dependencias en .venv...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
".venv\Scripts\python.exe" -m pip install -e . --no-deps --quiet

echo [3/3] Iniciando Ignait Studio...
start "" ".venv\Scripts\pythonw.exe" -m ignait.gui
