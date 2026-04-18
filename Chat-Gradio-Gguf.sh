#!/bin/bash
# Script: `./Chat-Gradio-Gguf.sh`

# Set terminal configuration (Linux/Unix method)
echo -ne "\033]0;Chat-Gradio-Gguf\007"
printf '\e[8;25;82t'

# Change to script dir#!/bin/bash
# Script: `./Chat-Gradio-Gguf.sh`

# Set terminal configuration (Linux/Unix method)
echo -ne "\033]0;Chat-Gradio-Gguf\007"
printf '\e[8;25;82t'

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
    echo "    1. Run Main Program"
    echo ""
    echo "    2. Run Installation"
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    display_separator_thick
    read -p "Selection; Menu Options = 1-2, Exit Bash = X: " choice
}

# Function to run main program
run_main_program() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Launcher"
    display_separator_thick
    echo ""
    echo "Starting Chat-Gradio-Gguf..."
    # Call the venv Python directly - no activate/deactivate needed.
    # Python resolves its own site-packages relative to its executable location,
    # so the venv works correctly even when the project folder has been moved.
    VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
    if [ ! -f "$VENV_PYTHON" ]; then
        echo "Error: Virtual environment not found at .venv/bin/python"
        echo "Run installation first (option 2)."
        read -p "Press Enter to continue..."
        return
    fi
    export PYTHONUNBUFFERED=1
    "$VENV_PYTHON" ./launcher.py linux
    if [ $? -ne 0 ]; then
        echo "Error launching Chat-Gradio-Gguf"
        read -p "Press Enter to continue..."
    fi
    unset PYTHONUNBUFFERED
}

# Function to run installation
run_installation() {
    clear
    display_separator_thick
    echo "    Chat-Gradio-Gguf: Installer"
    display_separator_thick
    echo ""

    python3 ./installer.py linux
    if [ $? -ne 0 ]; then
        echo "Error during installation"
        read -p "Press Enter to continue..."
    fi
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