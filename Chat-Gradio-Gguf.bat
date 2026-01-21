REM .\Chat-Gradio-Gguf.bat
@echo off
setlocal enabledelayedexpansion

REM ==== Static Configuration ====
set "TITLE=Chat-Gradio-Gguf"
title %TITLE%
mode con cols=80 lines=30
powershell -noprofile -command "& { $w = $Host.UI.RawUI; $b = $w.BufferSize; $b.Height = 6000; $w.BufferSize = $b; }"

:: DP0 TO SCRIPT BLOCK
set "ScriptDirectory=%~dp0"
set "ScriptDirectory=%ScriptDirectory:~0,-1%"
cd /d "%ScriptDirectory%"
echo Dp0'd to Script.

REM ==== Admin Check ====
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Error: Admin Required!
    timeout /t 2 >nul
    echo Right Click, Run As Administrator.
    timeout /t 2 >nul
    goto :end_of_script
)
echo Status: Administrator
timeout /t 1 >nul

REM ==== Function Definitions ====
goto :SkipFunctions

REM ==== Separator Functions ====
:DisplaySeparatorThick
echo ===============================================================================
goto :eof

:DisplaySeparatorThin
echo -------------------------------------------------------------------------------
goto :eof

REM ==== Main Menu ====
:MainMenu
cls
color 0F
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Batch Menu
call :DisplaySeparatorThick
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo     1. Run Main Program
echo.
echo     2. Run Installation
echo.
echo     3. Run Validation
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick
set /p "choice=Selection; Menu Options = 1-3, Exit Batch = X: "

if /i "%choice%"=="1" goto :RunMainProgram
if /i "%choice%"=="2" goto :RunInstallation
if /i "%choice%"=="3" goto :RunValidation
if /i "%choice%"=="X" goto :end_of_script

echo Invalid selection. Please try again.
timeout /t 2 >nul
goto :MainMenu

REM ==== Option 1: Run Main Program ====
:RunMainProgram
cls
color 06
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Launcher
call :DisplaySeparatorThick
echo.
echo Starting %TITLE%...
set PYTHONUNBUFFERED=1

call .\.venv\Scripts\activate.bat
echo Activated: `.venv`

python.exe -u .\launcher.py windows
if errorlevel 1 (
    echo Error launching %TITLE%
    pause
)
call .\.venv\Scripts\deactivate.bat
echo DeActivated: `.venv`
set PYTHONUNBUFFERED=0
goto :MainMenu

REM ==== Option 2: Run Installation ====
:RunInstallation
cls
color 06
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Installer
call :DisplaySeparatorThick
echo.
echo Running Installer...
timeout /t 1 >nul

rmdir /s /q .\data
echo Deleted: .\data
echo Foolproofing VENV Deletion
call .\.venv\Scripts\deactivate.bat
rmdir /s /q .\.venv
echo Deleted: .\.venv
echo.
echo Preparation Complete.
timeout /t 1 >nul
echo Running Installer...
timeout /t 3 >nul

cls
python.exe .\installer.py windows
if errorlevel 1 (
    echo Error during installation
    pause
)
call .\.venv\Scripts\deactivate.bat
echo DeActivated: `.venv`
set PYTHONUNBUFFERED=0
pause
goto :MainMenu

REM ==== Option 3: Run Validation ====
:RunValidation
cls
color 06
call :DisplaySeparatorThick
echo     Chat-Gradio-Gguf: Library Validation
call :DisplaySeparatorThick
echo.
echo Running Library Validation...

call .\.venv\Scripts\activate.bat
echo Activated: `.venv`

python.exe .\validater.py windows

call .\.venv\Scripts\deactivate.bat
echo DeActivated: `.venv`
pause
goto :MainMenu

:SkipFunctions

REM ==== Start at Main Menu ====
goto :MainMenu

:end_of_script
echo Closing %TITLE%...
timeout /t 2 >nul
exit