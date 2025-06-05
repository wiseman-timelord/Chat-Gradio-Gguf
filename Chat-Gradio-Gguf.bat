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

REM ==== 80-Column Version ====
:MainMenu80
cls
color 0F
call :DisplaySeparatorThick80
echo     Chat-Gradio-Gguf: Batch Menu
call :DisplaySeparatorThick80
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
echo.
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick80
set /p "choice=Selection; Menu Options = 1-2, Exit Batch = X: "
goto :ProcessChoice80

REM ==== 120-Column Version ====
:MainMenu120
cls
color 0F
call :DisplaySeparatorThick120
echo "                                  ___________      ________          ________                                        "
echo "                                  \__    ___/     /  _____/         /  _____/                                        "
echo "                                    |    | ______/   \  ___  ______/   \  ___                                        "
echo "                                    |    |/_____/\    \_\  \/_____/\    \_\  \                                       "
echo "                                    |____|        \______  /        \______  /                                       "
echo "                                                         \/                \/                                        "
call :DisplaySeparatorThin120
echo                                     Chat-Gradio-Gguf: Batch Menu                                      
call :DisplaySeparatorThick120
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo                                     1. Run Main Program
echo.
echo                                     2. Run Installation
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
call :DisplaySeparatorThick120
set /p "choice=Selection; Menu Options = 1-2, Exit Batch = X: "
goto :ProcessChoice120

REM ==== Common Processing ====
:ProcessChoice80
if /i "%choice%"=="1" (
    cls
    color 06
    call :DisplaySeparatorThick80
    echo     Chat-Gradio-Gguf: Launcher
    call :DisplaySeparatorThick80
    echo.
    echo Starting %TITLE%...
    set PYTHONUNBUFFERED=1
    
    REM [Rest of your processing code...]
    goto MainMenu80
)

if /i "%choice%"=="2" (
    cls
    color 06
    call :DisplaySeparatorThick80
    echo     Chat-Gradio-Gguf: Installer
    call :DisplaySeparatorThick80
    echo.
    echo Running Installer...
    REM [Rest of your processing code...]
    goto MainMenu80
)

if /i "%choice%"=="X" (
    cls
    echo Closing %TITLE%...
    timeout /t 2 >nul
    goto :end_of_script
)

echo Invalid selection. Please try again.
timeout /t 2 >nul
goto MainMenu80

:ProcessChoice120
if /i "%choice%"=="1" (
    cls
    color 06
    call :DisplaySeparatorThick120
    echo                                     Chat-Gradio-Gguf: Launcher                                      
    call :DisplaySeparatorThick120
    echo.
    echo Starting %TITLE%...
    set PYTHONUNBUFFERED=1
    
    REM [Rest of your processing code...]
    goto MainMenu120
)

if /i "%choice%"=="2" (
    cls
    color 06
    call :DisplaySeparatorThick120
    echo                                     Chat-Gradio-Gguf: Installer                                      
    call :DisplaySeparatorThick120
    echo.
    echo Running Installer...
    REM [Rest of your processing code...]
    goto MainMenu120
)

if /i "%choice%"=="X" (
    cls
    echo Closing %TITLE%...
    timeout /t 2 >nul
    goto :end_of_script
)

echo Invalid selection. Please try again.
timeout /t 2 >nul
goto MainMenu120

:SkipFunctions
REM ==== Auto-detect which menu to show ====
mode con | find "120" >nul
if %errorlevel%==0 (
    goto :MainMenu120
) else (
    goto :MainMenu80
)

:end_of_script
cls
color 0F
call :DisplaySeparatorThick80
echo. 
timeout /t 2 >nul
exit