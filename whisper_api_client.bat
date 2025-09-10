@echo off
REM Убедитесь, что путь к Python указан правильно
REM Если Python добавлен в PATH, то достаточно указать просто python

set PYTHON_PATH=D:/SKZ/venv/Scripts/python.exe
set SCRIPT_PATH="D:\Projects\whisper_test\whisper-obsidian-plugin\whisper_api_client.py"

echo Запуск Python скрипта...
"%PYTHON_PATH%" "%SCRIPT_PATH%"

REM Если хотите, чтобы окно оставалось открытым после выполнения:
pause
