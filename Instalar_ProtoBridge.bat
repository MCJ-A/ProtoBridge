@echo off
chcp 65001 >nul
title ProtoBridge Studio v1.0 - Instalador Oficial [By CRESKO // Jhon Córdova]
color 0B

echo ============================================================================
echo    PROTOBRIDGE STUDIO v1.0 - TRADUCTOR TACTICO FULL-DUPLEX EN TIEMPO REAL
echo    Autor: Jhon Córdova Martínez (Alias: Cresko)
echo    GitHub: https://github.com/MCJ-A/ProtoBridge
echo    LinkedIn: https://www.linkedin.com/in/jhon-cordova-martinez
echo ============================================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creando entorno virtual aislado Python (.venv)...
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)

echo [2/4] Instalando dependencias de baja latencia y CUDA...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet

echo [3/4] Verificando modelos IA neurales y driver virtual VB-Cable...
".venv\Scripts\python.exe" -m protobridge.auto_setup

echo [4/4] Creando acceso directo "ProtoBridge Studio (by Cresko)" en el Escritorio...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\ProtoBridge Studio (by Cresko).lnk'); $s.TargetPath = '%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments = '-m protobridge.gui'; $s.WorkingDirectory = '%~dp0'; $s.Description = 'ProtoBridge Studio v1.0 - Desarrollado por Jhon Córdova (Cresko)'; $s.Save()"

echo.
echo [LISTO] Instalacion completada. Iniciando ProtoBridge Studio...
start "" ".venv\Scripts\pythonw.exe" -m protobridge.gui
