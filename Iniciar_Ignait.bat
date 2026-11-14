@echo off
title Ignait Studio
cd /d "%~dp0"
taskkill /F /IM pythonw.exe >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
start "" "%~dp0.venv\Scripts\pythonw.exe" -m ignait.gui
