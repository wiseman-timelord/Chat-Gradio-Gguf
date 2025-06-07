REM .\Chat-Gradio-Gguf.bat
@echo off
setlocal enabledelayedexpansion

REM ==== Static Configuration ====
set "TITLE=Chat-Gradio-Gguf"
title %TITLE%

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

REM ==== Detect Terminal Width ====
:DetectWidth
mode con | find "120" >nul
if %errorlevel%==0 (
    set "WIDE_MODE=1"
) else (
    set "WIDE_MODE=0"
)
goto :eof

REM ==== Display Functions ====
:DisplaySeparator
if "%WIDE_MODE%"=="1" (
    echo =======================================================================================================================
) else (
    echo ===============================================================================
)
goto :eof

REM ==== Main Menu ====
:MainMenu
cls
color 0F
call :DetectWidth
call :DisplaySeparator
echo     Chat-Gradio-Gguf: Batch Menu
call :DisplaySeparator
echo.
echo Menu Options:
echo     1. Run Main Program
echo     2. Run Installation
echo     3. Run Validation
echo     X. Exit Batch Menu
echo.
set /p "choice=Selection; Options = 1-3, Exit = X: "
goto :ProcessChoice

REM ==== Choice Processing ====
:ProcessChoice
if /i "%choice%"=="1" (
    cls
    color 06
    call :DisplaySeparator
    echo     Chat-Gradio-Gguf: Launcher
    call :DisplaySeparator
    echo.
    echo Starting %TITLE%...
    set PYTHONUNBUFFERED=1
    
    call .\.venv\Scripts\activate.bat
    echo Activated: `.venv`
	
    python.exe -u .\launcher.py
    if errorlevel 1 (
        echo Error launching %TITLE%
        pause
    )
    call .\.venv\Scripts\deactivate.bat
    echo DeActivated: `.venv`
    set PYTHONUNBUFFERED=0
    goto MainMenu
)

if /i "%choice%"=="2" (
    cls
    color 06
    call :DisplaySeparator
    echo     Chat-Gradio-Gguf: Installer
    call :DisplaySeparator
    echo.
    echo Running Installer...
    timeout /t 1 >nul
    
    call .\.venv\Scripts\activate.bat
    echo Activated: `.venv`
    
    cls
    python.exe .\installer.py installer
    if errorlevel 1 (
        echo Error during installation
        pause
    )
    call .\.venv\Scripts\deactivate.bat
    echo DeActivated: `.venv`
    set PYTHONUNBUFFERED=0
    pause
    goto MainMenu
)

if /i "%choice%"=="3" (
    cls
    color 06
    call :DisplaySeparater
    echo     Chat-Gradio-Gguf: Library Validation
    call :DisplaySeparator
    echo.
    echo Running Library Validation...
    
    call .\.venv\Scripts\activate.bat
    echo Activated: `.venv`

    python.exe .\validater.py

    call .\.venv\Scripts\deactivate.bat
    echo DeActivated: `.venv`
    pause
    goto MainMenu
)

if /i "%choice%"=="X" (
    goto :end_of_script
)

echo Invalid selection. Please try again.
timeout /t 2 >nul
goto MainMenu

:SkipFunctions
REM ==== Start Main Menu ====
goto :MainMenu

:end_of_script
echo Closing %TITLE%...
timeout /t 2 >nul
exit