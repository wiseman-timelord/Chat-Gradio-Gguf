REM .\Chat-Gradio-Gguf.bat
@echo off
setlocal enabledelayedexpansion

REM display setup
REM mode con: cols=80 lines=25

REM title code
set "TITLE=Chat-Gradio-Gguf"
title %TITLE%

:: DP0 TO SCRIPT BLOCK, DO NOT, MODIFY or MOVE: START
set "ScriptDirectory=%~dp0"
set "ScriptDirectory=%ScriptDirectory:~0,-1%"
cd /d "%ScriptDirectory%"
echo Dp0'd to Script.
:: DP0 TO SCRIPT BLOCK, DO NOT, MODIFY or MOVE: END

REM Check for Administrator privileges
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

REM Functions
goto :SkipFunctions

:DisplaySeparatorThick
echo =======================================================================================================================
goto :eof

:DisplaySeparatorThin
echo -----------------------------------------------------------------------------------------------------------------------
goto :eof

:DisplayTitle
call :DisplaySeparatorThick
echo "                                  ___________      ________          ________                                        "
echo "                                  \__    ___/     /  _____/         /  _____/                                        "
echo "                                    |    | ______/   \  ___  ______/   \  ___                                        "
echo "                                    |    |/_____/\    \_\  \/_____/\    \_\  \                                       "
echo "                                    |____|        \______  /        \______  /                                       "
echo "                                                         \/                \/                                        "
call :DisplaySeparatorThin
goto :eof

call :DisplaySeparatorThin
goto :eof

:MainMenu
cls
color 0F
call :DisplayTitle
echo     Chat-Gradio-Gguf: Batch Menu
call :DisplaySeparatorThick
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
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick
set /p "choice=Selection; Menu Options = 1-2, Exit Batch = X: "

REM Process user input
if /i "%choice%"=="1" (
    cls
    color 06
    call :DisplaySeparatorThick
    echo     Chat-Gradio-Gguf: Launcher
    call :DisplaySeparatorThick
    echo.
    echo Starting %TITLE%...
    set PYTHONUNBUFFERED=1
    
    REM Activate venv and launch
    call .\.venv\Scripts\activate.bat
    python.exe .\requisites.py testlibs
	python.exe -u .\launcher.py
    
    REM Check for errors
    if errorlevel 1 (
        echo Error launching %TITLE%
        pause
    )
    
    REM Deactivate venv using full path to batch file
    call .\.venv\Scripts\deactivate.bat
    set PYTHONUNBUFFERED=0
    goto MainMenu
)

if /i "%choice%"=="2" (
    cls
    color 06
    call :DisplaySeparatorThick
    echo     Chat-Gradio-Gguf: Installer
    call :DisplaySeparatorThick
    echo.
    echo Running Installer...
    timeout /t 1 >nul
    cls
    python.exe .\requisites.py installer
    if errorlevel 1 (
        echo Error during installation
    )
    call .\.venv\Scripts\deactivate.bat
    set PYTHONUNBUFFERED=0
    pause
    goto MainMenu
)

if /i "%choice%"=="X" (
    cls
    echo Closing %TITLE%...
    timeout /t 2 >nul
    goto :end_of_script
)

REM Invalid input handling
echo Invalid selection. Please try again.
timeout /t 2 >nul
goto MainMenu

:SkipFunctions
goto MainMenu

:end_of_script
cls
color 0F
call :DisplayTitle
echo. 
timeout /t 2 >nul
exit