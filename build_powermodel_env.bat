@echo off
cd /d "%~dp0"

set "ENV_FILE=environment_powermodel.yml"
set "LOCAL_ENV=%CD%\.conda\envs\powermodel"

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] conda is not available in this terminal.
  echo Open Anaconda Prompt and run this script again.
  exit /b 1
)

if not exist "%ENV_FILE%" (
  echo [ERROR] %ENV_FILE% not found in %CD%
  exit /b 1
)

if exist "%LOCAL_ENV%" (
  echo Updating existing env at:
  echo   %LOCAL_ENV%
  call conda env update --file "%ENV_FILE%" --prefix "%LOCAL_ENV%" --prune
) else (
  echo Creating new env at:
  echo   %LOCAL_ENV%
  call conda env create --file "%ENV_FILE%" --prefix "%LOCAL_ENV%"
)

if errorlevel 1 (
  echo [ERROR] Conda environment build failed.
  exit /b 1
)

echo.
echo Environment ready.
echo Activate it with:
echo   conda activate "%LOCAL_ENV%"
