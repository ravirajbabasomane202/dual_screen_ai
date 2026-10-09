@echo off
REM Builds dist\DualScreenAI\DualScreenAI.exe, a portable zip, and (if Inno Setup is installed) an installer.
cd /d "%~dp0\.."
python -m pip install -r requirements-dev.txt || exit /b 1
pyinstaller build\DualScreenAI.spec --noconfirm || exit /b 1

REM Portable version: a 'portable.flag' file next to the exe keeps all data in .\data
type nul > dist\DualScreenAI\portable.flag
powershell -NoProfile -Command "Compress-Archive -Force dist\DualScreenAI\* dist\DualScreenAI-portable.zip"
del dist\DualScreenAI\portable.flag

where iscc >nul 2>nul && iscc build\installer.iss
echo.
echo Done. Output in dist\
