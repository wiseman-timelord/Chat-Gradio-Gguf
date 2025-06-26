REM .\Chat-Windows-Gguf.bat
@echo off
setlocal enabledelayedexpansion

REM ==== Static Configuration ====
set "TITLE=Chat-Windows-Gguf"
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

REM ==== Separator Functions ====
:DisplaySeparatorThick80
echo ===============================================================================
goto :eof

:DisplaySeparatorThin80
echo -------------------------------------------------------------------------------
goto :eof

:DisplaySeparatorThick120
echo =======================================================================================================================
goto :eof

:DisplaySeparatorThin120
echo ----------------------------------------------------------------------------------------------------------------------
goto :eof

REM ==== Dynamic Separator Functions ====
:DisplaySeparatorThick
if "%MENU_WIDTH%"=="120" (
    call :DisplaySeparatorThick120
) else (
    call :DisplaySeparatorThick80
)
goto :eof

:DisplaySeparatorThin
if "%MENU_WIDTH%"=="120" (
    call :DisplaySeparatorThin120
) else (
    call :DisplaySeparatorThin80
)
goto :eof

REM ==== 80-Column Version ====
:MainMenu80
cls
color 0F
call :DisplaySeparatorThick80
echo     Chat-Windows-Gguf: Batch Menu
call :DisplaySeparatorThick80
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
call :DisplaySeparatorThick80
set /p "choice=Selection; Menu Options = 1-3, Exit Batch = X: "
goto :ProcessChoice

REM ==== 120-Column Version ====
:MainMenu120
cls
color 0F
call :DisplaySeparatorThick120
echo "                                  _________             ________            ________                                 "
echo "                                  \_   ___ \           /  _____/           /  _____/                                 "
echo "                                  /    \  \/   ______ /   \  ___   ______ /   \  ___                                 "
echo "                                  \     \____ /_____/ \    \_\  \ /_____/ \    \_\  \                                "
echo "                                   \______  /          \______  /          \______  /                                "
echo "                                          \/                  \/                  \/                                 "
call :DisplaySeparatorThin120
echo    Chat-Windows-Gguf: Batch Menu                                      
call :DisplaySeparatorThick120
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo    1. Run Main Program
echo.
echo    2. Run Installation
echo.
echo    3. Run Validation
echo.
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick120
set /p "choice=Selection; Menu Options = 1-3, Exit Batch = X: "
goto :ProcessChoice

REM ==== Unified Choice Processing ====
:ProcessChoice
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
if "%MENU_WIDTH%"=="120" (
    echo                                     Chat-Windows-Gguf: Launcher                                      
) else (
    echo     Chat-Windows-Gguf: Launcher
)
call :DisplaySeparatorThick
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
goto :MainMenu

REM ==== Option 2: Run Installation ====
:RunInstallation
cls
color 06
call :DisplaySeparatorThick
if "%MENU_WIDTH%"=="120" (
    echo                                     Chat-Windows-Gguf: Installer                                      
) else (
    echo     Chat-Windows-Gguf: Installer
)
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
echo Preperation Complete.
timeout /t 1 >nul
echo Running Installer...
timeout /t 3 >nul

cls
python.exe .\installer.py
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
if "%MENU_WIDTH%"=="120" (
    echo                                     Chat-Windows-Gguf: Library Validation                                      
) else (
    echo     Chat-Windows-Gguf: Library Validation
)
call :DisplaySeparatorThick
echo.
echo Running Library Validation...

call .\.venv\Scripts\activate.bat
echo Activated: `.venv`

python.exe .\validater.py

call .\.venv\Scripts\deactivate.bat
echo DeActivated: `.venv`
pause
goto :MainMenu

REM ==== Main Menu Router ====
:MainMenu
if "%MENU_WIDTH%"=="120" (
    goto :MainMenu120
) else (
    goto :MainMenu80
)

:SkipFunctions
REM ==== Auto-detect which menu to show ====
mode con | find "120" >nul
if %errorlevel%==0 (
    set "MENU_WIDTH=120"
    goto :MainMenu120
) else (
    set "MENU_WIDTH=80"
    goto :MainMenu80
)

:end_of_script
echo Closing %TITLE%...
timeout /t 2 >nul
exit