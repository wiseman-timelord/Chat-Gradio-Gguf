#!/bin/bash
# Script: `./Chat-Gradio-Gguf.sh`

# Set terminal configuration (Linux/Unix method)
echo -ne "\033]0;Chat-Gradio-Gguf\007"
printf '\e[8;29;80t'

# Change to script directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR" || exit 1
echo "Changed to script directory: $SCRIPT_DIR"

# Check for root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: Root privileges required!"
    sleep 2
    echo "Run with sudo or as root."
    sleep 2
    exit 1
fi
echo "Status: Root"
sleep 1

# Separator functions
display_separator_thick() {
    echo "==============================================================================="
}

display_separator_thin() {
    echo "-------------------------------------------------------------------------------"
}

# Main menu
main_menu() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Bash Menu"
    display_separator_thick
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "    1. Run Main Program"
    echo ""
    echo "    2. Run Installation"
    echo ""
    echo "    3. Run Validation"
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    display_separator_thick
    read -p "Selection; Menu Options = 1-3, Exit Bash = X: " choice
}

# Function to run main program
run_main_program() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Launcher"
    display_separator_thick
    echo ""
    echo "Starting Chat-Gradio-Gguf..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Activated: .venv"
    else
        echo "Error: Virtual environment (.venv) not found"
        read -p "Press Enter to continue..."
        return
    fi
    export PYTHONUNBUFFERED=1
    python3 ./launcher.py linux
    if [ $? -ne 0 ]; then
        echo "Error launching Chat-Gradio-Gguf"
        read -p "Press Enter to continue..."
    fi
    deactivate
    echo "Deactivated: .venv"
    unset PYTHONUNBUFFERED
}

# Function to run installation
run_installation() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Installer"
    display_separator_thick
    echo ""
    echo "Running Installer..."
    sleep 1
    rm -rf ./data
    echo "Deleted: ./data"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        deactivate
        rm -rf .venv
        echo "Deleted: .venv"
    else
        echo "No existing .venv to delete"
    fi
    echo ""
    echo "Preparation Complete."
    sleep 1
    echo "Running Installer..."
    sleep 3
    clear
    python3 ./installer.py linux
    if [ $? -ne 0 ]; then
        echo "Error during installation"
        read -p "Press Enter to continue..."
    fi
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        deactivate
        echo "Deactivated: .venv"
        read -p "Press Enter to continue..."
    fi
}

# Function to run validation
run_validation() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Library Validation"
    display_separator_thick
    echo ""
    echo "Running Library Validation..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "Activated: .venv"
    else
        echo "Error: Virtual environment (.venv) not found"
        read -p "Press Enter to continue..."
        return
    fi
    python3 ./validater.py linux
    if [ $? -ne 0 ]; then
        echo "Error during validation"
        read -p "Press Enter to continue..."
    fi
    deactivate
    echo "Deactivated: .venv"
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    main_menu
    case $choice in
        1)
            run_main_program
            ;;
        2)
            run_installation
            ;;
        3)
            run_validation
            ;;
        X|x)
            echo "Closing Chat-Gradio-Gguf..."
            sleep 2
            exit 0
            ;;
        *)
            echo "Invalid selection. Please try again."
            sleep 2
            ;;
    esac
done