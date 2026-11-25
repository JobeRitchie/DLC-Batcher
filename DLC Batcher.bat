@echo off
rem Starts DLC Batcher with the Python of your DeepLabCut environment.
rem Order: DLC_BATCHER_PYTHON, the environment saved by the app, a search of
rem conda environments and the usual venv folders, then asking for a path.
setlocal EnableExtensions
cd /d "%~dp0"

set "PY="
set "BOOT="

if defined DLC_BATCHER_PYTHON (
  if exist "%DLC_BATCHER_PYTHON%" (
    set "PY=%DLC_BATCHER_PYTHON%"
    goto run
  )
  echo DLC_BATCHER_PYTHON points to "%DLC_BATCHER_PYTHON%", which does not exist. Searching instead.
  set "DLC_BATCHER_PYTHON="
)

rem the environment saved in settings.json
set "CFG=%UserProfile%\.dlc_batcher\settings.json"
if defined DLC_BATCHER_HOME set "CFG=%DLC_BATCHER_HOME%\settings.json"
if exist "%CFG%" (
  for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "try { (Get-Content -Raw -LiteralPath $env:CFG | ConvertFrom-Json).python_exe } catch { }"`) do (
    if exist "%%P" set "BOOT=%%P"
  )
)
if defined BOOT goto resolve

rem any environment with DeepLabCut will do to run the full search
for %%V in ("%UserProfile%\Desktop\DeepLabCut\venv" "%UserProfile%\Desktop\DeepLabCut\.venv" "%UserProfile%\DeepLabCut\venv" "%UserProfile%\DeepLabCut\.venv" "%~dp0.venv" "%~dp0venv") do (
  if not defined BOOT if exist "%%~V\Scripts\python.exe" if exist "%%~V\Lib\site-packages\deeplabcut-*.dist-info" set "BOOT=%%~V\Scripts\python.exe"
)
for %%R in ("%UserProfile%\anaconda3" "%UserProfile%\miniconda3" "%UserProfile%\miniforge3" "%UserProfile%\mambaforge" "%UserProfile%\.conda" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3") do (
  if not defined BOOT if exist "%%~R\python.exe" if exist "%%~R\Lib\site-packages\deeplabcut-*.dist-info" set "BOOT=%%~R\python.exe"
  if exist "%%~R\envs" for /d %%E in ("%%~R\envs\*") do (
    if not defined BOOT if exist "%%E\python.exe" if exist "%%E\Lib\site-packages\deeplabcut-*.dist-info" set "BOOT=%%E\python.exe"
  )
)
if not defined BOOT goto ask

:resolve
rem let the app's own search pick: saved choice first, otherwise the newest DeepLabCut
for /f "usebackq delims=" %%P in (`""%BOOT%" -m dlc_batcher.dlc_env --resolve"`) do set "PY=%%P"
if not defined PY set "PY=%BOOT%"
goto run

:ask
echo.
echo No Python environment with DeepLabCut was found.
echo Paste the full path to the python.exe of your DeepLabCut environment
echo (for a venv this is ...\venv\Scripts\python.exe), or press Enter to quit.
set "PICK="
set /p "PICK=python.exe: "
if not defined PICK exit /b 1
set "PICK=%PICK:"=%"
if not exist "%PICK%" (
  echo "%PICK%" does not exist.
  goto ask
)
set "PY="
for /f "usebackq delims=" %%P in (`""%PICK%" -m dlc_batcher.dlc_env --save "%PICK%""`) do set "PY=%%P"
if not exist "%PY%" (
  echo %PY%
  goto ask
)

:run
echo Starting DLC Batcher with %PY%
"%PY%" -m dlc_batcher
if errorlevel 1 pause
