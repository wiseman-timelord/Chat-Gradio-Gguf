# ![Chat-Windows-Gguf](media/project_banner.jpg)

### Status
Beta - Latest release v1.04.x now includes "About/Debug Info" tab with contact/support info. See release notes.

## Description
Intended as a high-quality chat interface with wide hardware/os support, windows 7-11 (WSL not required) and Ubuntu 22-25, with any Gpu on GGUF models through Python ~3.9-3.13. An optimal number of features for a ChatBot, as well as, dynamic buttons/panels on the interface and websearch and RAG and TTS and archiving of sessions, and all on local models, so no imposed, limitations or guidelines (model dependent). This tool providing a comparable interface found on premium non-agentic AI services, where the configuration is intended to be intelligent, while without options reported in forums to make no difference on most models (no over-complication). The program using offline libraries (apart from websearch) instead of, online services or repeat download or registration. One thing to mention though, is that because a goal of the project is to make a Chatbot compatible as it is, therefore compromises had to be made, and it may not have all the latest versions of things, more the most widely compatible versions of things, so some features are not as good as they could be, but also, this program serves as a valuable reference for what needs to be used on what OS, albeit it, as a personal project it could be streamlined/enhanced/customized to your own likings later.   

### Core Principles
- This project is for a chat interface, and is not intended to overlap with other blueprints/projects, `Rpg-Gradio-Gguf` or `Code-Gradio-Gguf` or `Agent-Gradio-Gguf`. 
- A, Windows 7-11 and Ubuntu 22-25, compatibility range; though Ubuntu 25 users have limitations I believe already due to python 3.13. Any and all compatibility issues within those ranges MUST be overcome.
- This Program is also intended to have only basic tools of, DDG Search, Web Search, and Speech Out. The cost of having wide compatibility is scrifice of tool complexity.
- I am making this program for the community, so that people on Ubuntu and Windows, are able to use a single chatbot for AI, even on legacy setups or modern systems.

### Features
- **Qt-Web Custom Browser**: The interface uses Qt-Web with Gradio, it appears as a regular application, and means your default browser are untouched.  
- **Comprihensive GPU Support**: Vulkan, with dropdown list in configuration selection supporting multi CPU/GPU setup.
- **Research-Grade Tools**: Includes RAG, web search, chunking, THINK, and Markdown formatting, and file attachments. 
- **Text To Speech**: Using built-in Windows/Ubuntu speech capabilities, for basic TTS tool.
- **Common File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`, and other common formats of files.
- **Configurable Context**: Set model context to 8192-138072, and batch output to 256-8192.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Highly Customizable UI**: Configurable; 4-16 Session History slots, 2-10 file slots, Session Log 450-1300px height, 2-8 Lines of input. 
- **Speak Summaries**: Click `Speech` for a special prompt for a concise spoken summary of the generated output. Text to speak uses `PyWin32`.
- **Attach Files**: Attach Files is complete raw input, there is no Vectorise Files anymore, so the files should ideally be small enough to fit in context. 
- **Collapsable Left/Right Column**: Like one would find on modern AI interface, and with concise collapsed view interface for commonly used buttons. 
- **ASynchronous Response Stream**: Separate thread with its own event loop, allowing chunks of response queued/processed without blocking Gradio UI event loop.
- **Reasoning Compatible**: Dynamic prompt system adapts handling for reasoning models optimally, ie, uncensored, nsfw, chat, code.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Fast and Optimised**: Optionally compiling Vulkan backend/wheel with special AVX/FMA/F16C optimisations, as well as runtime optimizations for vulkan.

### Preview
- The Interaction page on Windows, showing the Web Search tool, with Windows Command Prompt in the background (v1.02.1)...
![image_missing](media/conversation_page_windows.jpg)

- The Interaction page on Ubuntu, here showing the DDG Search tool, with Ubuntu Terminal in the background (v1.02.1)...
![image_missing](media/conversation_page_ubuntu.jpg)

- The Config page, for configuration of, Hardware, TTS, Models (v1.02.1)...
![image_missing](media/config_page.jpg)

- The Settings page, for configuration of, GUI, Filters (v1.02.1)...
![image_missing](media/settings_page.jpg)

- The About/Debug Info page, displaying useful information (v1.04.0)...
![image_missing](media/about_page.jpg)

- The collapseable Left/Right Panel on the Interaction page (click the `<->` button)...
![image_missing](media/conversation_expand.jpg)

- The dynamic progress indication replaces the user input box upon submission...
![image_missing](media/dynamic_progress.jpg)

- Startup looks like this in the terminal/console (ignore warnings) (v1.02.1)...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf: Launcher
    ===============================================================================
    
    Starting Chat-Gradio-Gguf...
    Activated: `.venv`
    C:\LocalAI_Files\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.02.1\.venv\Lib\site-package
    s\transformers\utils\hub.py:124: FutureWarning: Using `TRANSFORMERS_CACHE` is de
    precated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
      warnings.warn(
    `main` Function Started.
    [INI] Platform: windows
    [INI] Backend: VULKAN_CPU
    [INI] Vulkan: True
    [INI] Embedding Model: BAAI/bge-base-en-v1.5
    [INI] Gradio Version: 3.50.2
    [INI] OS Version: 8.1
    [INI] Windows Version: 8.1
    [INI] TTS section not found - will detect at runtime
    [TTS] Engine: pyttsx3
    [TTS] Audio Backend: windows
    [TTS] Default Voice set to: Microsoft David  - English (United States) (en-US)
    [MODELS] Scanning directory: models
    [MODELS] ? No .gguf files found in models
    [CONFIG] Auto-selected secondary GPU: Radeon (TM) RX 470 Graphics
    [CONFIG] Loaded -> Model: Select_a_model... | CPU: Auto-Select
    Configuration loaded
    [TTS] Engine: pyttsx3
    [TTS] Audio Backend: windows
    [TTS] Default Voice set to: Microsoft David  - English (United States) (en-US)
    [Vulkan] GGML_CUDA_NO_PINNED=1   (frees ~300 MB VRAM)
    [Vulkan] GGML_VK_NO_PIPELINE_CACHE=0  (cached SPIR-V pipelines)
    Script mode `windows` with backend `VULKAN_CPU`
    Working directory: ...les\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.02.1
    Data Directory: ...hat-Gradio-Gguf\Chat-Gradio-Gguf-1.02.1\data
    Session History: ...io-Gguf\Chat-Gradio-Gguf-1.02.1\data\history
    Temp Directory: ...radio-Gguf\Chat-Gradio-Gguf-1.02.1\data\temp
    [CPU] Detected: 12 cores, 24 threads
    [CPU] Current: 20
    CPU Configuration: 12 physical cores, 24 logical cores
    
    Configuration:
      Backend: VULKAN_CPU
      Model: Select_a_model...
      Context Size: 8192
      VRAM Allocation: 8192 MB
      CPU Threads: 20
      GPU Layers: 0
    
    [INIT] Pre-loading auxiliary inference...
    [INIT] OK spaCy model pre-loaded
    [RAG] Loading embedding model: BAAI/bge-base-en-v1.5
    C:\LocalAI_Files\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.02.1\.venv\Lib\site-package
    s\huggingface_hub\file_download.py:942: FutureWarning: `resume_download` is depr
    ecated and will be removed in version 1.0.0. Downloads always resume when possib
    le. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    [RAG] Embedding model loaded successfully
    [INIT] OK Embedding model pre-loaded from cache
    
    Launching Gradio display...
    [FILTER] Using gradio3 filter (15 rules)
    [FILTER] Using gradio3 filter (15 rules)
    [MODELS] Scanning directory: models
    [MODELS] ? No .gguf files found in models
    [BROWSER] Starting Gradio server in background...
    [BROWSER] Waiting for Gradio server at http://localhost:7860...
    Running on local URL:  http://localhost:7860
    
    To create a public link, set `share=True` in `launch()`.
    [BROWSER] Gradio server is ready
    [BROWSER] Launching at http://localhost:7860/?__theme=dark
    [BROWSER] Platform: windows, Windows Version: 8.1
    [BROWSER] Windows 8.1 detected - using Qt5 WebEngine
    [BROWSER] Loading URL: http://localhost:7860/?__theme=dark
    [BROWSER] Qt5 WebEngine window created
    [MODELS] Scanning directory: models
    [MODELS] ? No .gguf files found in models
    [MODEL] Dropdown updated | choices=1 models | selected=Select_a_model...

</details>

- The combined Info/Install menu  (v1.02.1)...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf - Configure Installation
    ===============================================================================
    
    System Detections...
        CPU Features: AVX, AVX2, FMA, F16C, SSE3, SSSE3, SSE4_1, SSE4_2
        Operating System: Windows 8.1
        Optimal Python: 3.11; Vulkan Present: Yes
        Build Tools: Git, CMake, MSVC, MSBuild
    
    Backend Options...
       1) Download CPU Wheel / Default CPU Wheel
       2) Download Vulkan Bin / Default CPU Wheel
       3) Compile CPU Binaries / Compile CPU Wheel
       4) Compile Vulkan Binaries / Compile Vulkan Wheel
    
    Embedding Model...
       a) Bge-Small-En v1.5 (Fastest - 132MB)
       b) Bge-Base-En v1.5 (Regular - 425MB)
       c) Bge-Large-En v1.5 (Quality - 1.35GB)
    
    ===============================================================================
    Selection; Backend=1-4, Embed=a-c, Abandon=A; (e.g. 2b):

</details>

- The TTS Install menu  (v1.03.0)...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf - TTS Configuration
    ===============================================================================
    
    TTS Engine...
       1) Built-In TTS (pyttsx3/espeak-ng)
       2) Coqui TTS (Higher quality, larger download ~1.4GB)
    
    If Coqui TTS, Select Voice...
       a) British (Male/Female)
       b) American (Male/Female)
       c) Scottish
       d) Irish
       e) Indian
       f) Canadian
       g) South African
       h) Welsh
       i) Northern Irish
       j) Australian
       k) New Zealand
    
    ===============================================================================
    Selection; TTS=1-2, Voice=a-k, Abandon=A; (e.g. 1 or 2b):

</details>

- Install with Options 4c (compile, large), 2a (Corqui, British), on Ubuntu 25 (v1.03.2)
<details>

    ===============================================================================
        Chat-Gradio-Gguf - Installation
    ===============================================================================
    
    Installing Chat-Gradio-Gguf on Ubuntu 25.04 with Python 3.13
      Route: Compile Vulkan Binaries / Compile Vulkan Wheel
      Llama.Cpp b7688, Gradio 5.49.1, Qt-Web v6
      Embedding: BAAI/bge-large-en-v1.5
      TTS: Coqui - British (Male/Female)
    [✓] Verified directory: data
    [✓] Verified directory: scripts
    [✓] Verified directory: models
    [✓] Verified directory: data/history
    [✓] Verified directory: data/temp
    [✓] Verified directory: data/vectors
    [✓] Verified directory: data/embedding_cache
    [✓] Using project temp path: /home/mastar/Programs/Chat-Gradio-Gguf/Chat-Gradio-Gguf-1.03.0/data/temp
    [✓] System information file created
    [✓] Installing Linux system dependencies...
    Hit:1 http://gb.archive.ubuntu.com/ubuntu plucky InRelease                     
    Hit:2 http://gb.archive.ubuntu.com/ubuntu plucky-updates InRelease             
    Hit:3 http://security.ubuntu.com/ubuntu plucky-security InRelease
    Hit:4 http://gb.archive.ubuntu.com/ubuntu plucky-backports InRelease  
    Hit:5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64  InRelease
    Reading package lists... Done                                   
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    xvfb is already the newest version (2:21.1.16-1ubuntu1.2).
    libespeak-ng1 is already the newest version (1.52.0+dfsg-4).
    libegl1 is already the newest version (1.7.0-1build1).
    libxkbcommon0 is already the newest version (1.7.0-2).
    build-essential is already the newest version (12.12ubuntu1).
    espeak-ng is already the newest version (1.52.0+dfsg-4).
    libgl1 is already the newest version (1.7.0-1build1).
    libffi-dev is already the newest version (3.4.7-1).
    libssl-dev is already the newest version (3.4.1-1ubuntu4).
    portaudio19-dev is already the newest version (19.6.0-1.2build3).
    libxcb-cursor0 is already the newest version (0.1.5-1).
    python3-dev is already the newest version (3.13.3-1).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓] Base dependencies installed
    [✓] Installing Qt6 dependencies...
    [✓]   Installed libxcb-cursor0
    [✓]   Installed libxkbcommon0
    [✓]   Installed libegl1
    [✓]   Installed libgl1
    [✓]   Installed libxcb-xinerama0
    [✓]   Installed qt6-base-dev
    [✓]   Installed libqt6webenginecore6
    [✓]   Installed libqt6webenginewidgets6
    [✓] Installing Vulkan development packages...
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    vulkan-tools is already the newest version (1.4.304.0+dfsg1-1).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓]   Installed vulkan-tools
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    libvulkan-dev is already the newest version (1.4.304.0-1).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓]   Installed libvulkan-dev
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    mesa-utils is already the newest version (9.0.0-2).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓]   Installed mesa-utils
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    glslang-tools is already the newest version (15.1.0-2).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓]   Installed glslang-tools
    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    spirv-tools is already the newest version (2025.1~rc1-1).
    Solving dependencies... Done
    The following packages were automatically installed and are no longer required:
      linux-headers-6.14.0-23 linux-headers-6.14.0-23-generic
      linux-image-6.14.0-23-generic linux-modules-6.14.0-23-generic
      linux-modules-extra-6.14.0-23-generic
      linux-modules-nvidia-570-6.14.0-23-generic
      linux-objects-nvidia-570-6.14.0-23-generic
      linux-signatures-nvidia-6.14.0-23-generic linux-tools-6.14.0-23
      linux-tools-6.14.0-23-generic
    Use 'sudo apt autoremove' to remove them.
    0 upgraded, 0 newly installed, 0 to remove and 347 not upgraded.
    [✓]   Installed spirv-tools
    [✓] Linux dependencies installed
    [✓] Installing Python dependencies...
    [✓] Installing base packages (Gradio 5.49.1)...
      [1/18] Installing numpy<2... OK
      [2/18] Installing requests... OK
      [3/18] Installing pyperclip... OK
      [4/18] Installing spacy... OK
      [5/18] Installing psutil... OK
      [6/18] Installing ddgs... OK
      [7/18] Installing langchain-community... OK
      [8/18] Installing faiss-cpu... OK
      [9/18] Installing langchain... OK
      [10/18] Installing pygments... OK
      [11/18] Installing lxml... OK
      [12/18] Installing lxml_html_clean... OK
      [13/18] Installing tokenizers... OK
      [14/18] Installing beautifulsoup4... OK
      [15/18] Installing aiohttp... OK
      [16/18] Installing pyvirtualdisplay... OK
      [17/18] Installing newspaper4k... OK
      [18/18] Installing gradio... OK
    [✓] Base packages installed
    [✓] Installing embedding backend (torch + sentence-transformers)...
    [✓] Installing PyTorch 2.4+ (CPU-only) for Python 3.13...
    [✓] PyTorch 2.4+ (CPU) installed
    [✓] Installing transformers>=4.42.0...
    [✓] transformers installed
    [✓] Installing sentence-transformers>=3.0.0...
    [✓] sentence-transformers installed
    [✓] Embedding backend verified
    [✓] Installing Qt WebEngine for custom browser...
    [✓] Ubuntu 25.04 - installing PyQt6 + Qt6 WebEngine...
    [✓] Qt6 WebEngine installed
    [✓] Vulkan wheel build - checking Vulkan SDK...
    [✓] Vulkan SDK detected
    [✓] Building llama-cpp-python from source (10-20 minutes)
      Using 20 parallel build threads
      Build flags: GGML_VULKAN=1, GGML_AVX2=ON, GGML_AVX=ON, GGML_FMA=ON, GGML_F16C=ON, LLAMA_CURL=OFF, GGML_OPENMP=ON
    [✓] Cloning llama-cpp-python v0.3.16...
    [✓] Building wheel (this takes a while)...
    (...)
    
    (...)
    [✓] llama-cpp-python built and installed
    [✓] Python dependencies installed successfully
    [✓] Installing optional file format support...
    [✓]   Installed PyPDF2
    [✓]   Installed python-docx
    [✓]   Installed openpyxl
    [✓]   Installed python-pptx
    [✓] Initializing embedding cache for BAAI/bge-large-en-v1.5...
    Embedding Initialization Output...
        Importing torch...
        torch version: 2.10.0+cpu
        CUDA available: False (should be False)
        Importing sentence_transformers...
        Loading model: BAAI/bge-large-en-v1.5
        Testing embedding...
        SUCCESS: Model loaded, dimension: 1024
    [✓] Embedding cache initialized
    [✓] Downloading spaCy language model...
    Downloading spaCy model: [==============================] 100% (12.2MB/12.2MB) - Complete
    [✓] Installing spaCy model...
    [✓] spaCy model installed
    [✓] Installing Coqui TTS with codec support...
    [✓] Installing torchaudio (CPU-only to match torch)...
    [✓] torchaudio (CPU) installed
    [✓] Coqui TTS package installed
    [✓] Downloading Coqui VCTK voice model (~1.4GB)...
    [COQUI] espeak-ng lib: /lib/x86_64-linux-gnu/libespeak-ng.so.1
    [COQUI] espeak-ng exe: /usr/bin/espeak-ng
    [COQUI] espeak-ng data: /usr/lib/x86_64-linux-gnu/espeak-ng-data
    [COQUI] Loading model...
    100%|███████████████████████████████████████| 148M/148M [00:28<00:00, 5.13MiB/s]
    [COQUI] Testing synthesis...
    [COQUI] Test passed
    [✓] Coqui TTS installed and verified
    [✓] Compiling llama.cpp binaries from source (15-30 minutes)...
      Building with 20 of 24 threads (85%)
    [✓] Cloning llama.cpp repository...
      Resolving deltas: 100% (508/508), done..38 MiB | 3.91 MiB/s, done.
    [✓] Repository cloned
    [✓] Configuring build...
    [✓] Building binaries...
      [100%] Linking CXX executable ../../bin/llama-serverlama-server.dir/se
    [✓] Copying binaries...
    [✓] llama.cpp binaries compiled successfully
    [✓] Configuration file created
    [✓] Installation complete!
    
    Run the launcher to start Chat-Gradio-Gguf
    
    Deactivated: .venv
    Press Enter to continue...

</details>

- Validation of Installation (v1.02.2)..
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf: Library Validation
    ===============================================================================
    
    Running Library Validation...
    Activated: `.venv`
    ==================================================
      Chat-Gradio-Gguf Validator (WINDOWS)
    ==================================================
    
    System: windows 8.1
    Python: 3.11.0 (Expected: 3.11.0)
    Backend: VULKAN_CPU
    Embedding: BAAI/bge-base-en-v1.5
    Gradio: 3.50.2, Qt: 5
    
    === Directory Validation ===
      V data (writable)
      V scripts (writable)
      V models (writable)
      V data/history (writable)
      V data/temp (writable)
      V data/vectors (writable)
      V data/embedding_cache (writable)
    
    === Root Files Validation ===
      V launcher.py
    
    === Script Files Validation ===
      V browser.py
      V display.py
      V inference.py
      V tools.py
      V configuration.py
      V utility.py
    
    === Configuration Validation ===
      V constants.ini valid (Backend: VULKAN_CPU)
      V persistent.json valid
    
    === Virtual Environment Validation ===
      V Python version matches: 3.11.0
      V Virtual environment OK
    
    === Vulkan Availability Check ===
      V Vulkan runtime found in registry
    
    === Backend Binary Validation ===
      V llama-cli found: llama-cli.exe
    
    === Core Library Validation ===
      V gradio
      V numpy
      V requests
      V pyperclip
      V spacy
      V psutil
      V ddgs
      V langchain-community
      V faiss-cpu
      V langchain
      V pygments
      V lxml
      V lxml_html_clean
      V tokenizers
      V beautifulsoup4
      V aiohttp
      V py-cpuinfo
      V sentence-transformers
      V torch
      V transformers
      V llama-cpp-python
      V newspaper4k
      V pywin32
      V tk
      V pythonnet
      V Gradio version matches: 3.50.2
    
    === Optional Library Validation ===
      V PyPDF2 (optional)
      V python-docx (optional)
      V openpyxl (optional)
      V python-pptx (optional)
    
    === spaCy Model Validation ===
      V en_core_web_sm model available
    
    === Embedding Model Validation ===
      V Embedding model verified (BAAI/bge-base-en-v1.5)
      V Backend: sentence_transformers
    
    === Gradio & Custom Browser Check ===
      Expected: Qt5 (PyQt5 WebEngine)
      V Gradio import OK
      V PyQt5 WebEngine import OK
    
    ==================================================
      Validation Summary
    ==================================================
      directories: V PASS
      root_files: V PASS
      scripts: V PASS
      config: V PASS
      venv: V PASS
      vulkan: V PASS
      llama_cli: V PASS
      core_libs: V PASS
      spacy: V PASS
      embedding: V PASS
      browser: V PASS
    
      Result: 11/11 checks passed
      V
    All validations passed!
      Installation is complete and ready to use.
    
    Press Enter to exit...
        
</details>

## Hard Requirements
- Windows 7-11 and/or ~Ubuntu 22-25 - Its BOTH a, Windows AND linux, program, batch for windows and bash for linux, launch dual-mode scripts.
- Python 3.9-3.13 - Requires [Python](https://www.python.org); AI warns me certain libraries wont work on Python 3.14, possibly Spacy. 
- Llm Model - You will need a Large Language Model in GGUF format, check the models section for recommendations, but for quick start I advise one like [Qwen3-4B-abliterated-GGUF](https://huggingface.co/mradermacher/Qwen3-4B-abliterated-GGUF) for testing basic operation.
- Suitable GPU - Gpu may be, Main or Compute, with VRam selection 4GB-64GB. It must have Vulkan capability/drivers. Ideally you want the GPU to cover all model layers for fast interference.
- System Ram - Your system ram must cover, the curren load and the size of the model layers not able to be covered by the GPU, plus smaller models like the embeddings model, plus the wheel if not built for Vulkan.

### Graphics Requirements
GPU interference is done through (have a guess) Vulkan...
- Vulkan - For vulkan install options you must install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

### Building Requirements 
For compile options; If on PATH, ask AI how to check its on path, and as applicable fix...
- [MSVC++ 2017-2022](https://visualstudio.microsoft.com/vs/older-downloads/) - MSVC with option, Desktop Development and CMake, enabled during install.
- [Git](https://git-scm.com/install/) - Github Program for cloning the sources from github, ensure its on PATH.
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - Need the Vulkan SDK to build for Vulkan.

### Instructions (W = Windows, U = Ubuntu)...
- Installation...
```
1.W. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `C:\Programs\Chat-Windows-Gguf` or `C:\Program_Files\Chat-Windows-Gguf`, (better not to use spaces with python projects). 
1.U. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `/media/**User_Name**/**Drive_Name**/Programs/Chat-Gradio-Gguf`, (better not to use spaces with python projects).
2.W. Right click the file `Chat-Windows-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load, then select `2` from the Batch Menu, to begin installation. You will be prompted to select a, Inference and Embedding, config to install, which should be done based on your hardware, and then you are prompted to select a TTS config. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder. 
2.U. open a terminal in the created folder location, then make "Chat-Gradio-Gguf.sh" executable via `Right Mouse>Properties>Make Executable`. Then run `sudo bash ./Chat-Linux-Gguf.sh`, then select option `2. Run Installation` from the menu.
- There are now 4 install options, mainly download, only or compile, and, cpu or gpu; If you do NOT have the 3 build apps installed then select 1/2, if you do SO have build apps installed then select 3/4. 
- The embedding model is the bit between your input and the model, larger is slower/quality obviously, and note its loaded only to system ram, because its ONNX. 
4. After the install completes, check for any install issues in the output, you may need to, check network or reinstall again, if there are issues. Pressing Enter will return you to the Batch Menu.
- For a first installI advise either, 2b then 2a (for modern machine, with english) or 1a then 1 (for old machine). If you have the build tools mentioned then on the first menu you should try 4a-2b (on a modern machine). 
```
- Running...
```
1. Having, returned to or run, the bash/batch menu, at some point after having run a successful install, one would use option `1. Run Main Program`, to load the gradio interface in the popup browser window. You will then be greeted with the `Interaction` page.
- You may, click maximise or drag to the side of the screen, to re-position window, as well as, press CTRL + WheelUp/WheelDown to resize the interface to your display size.
2. Yous should click the `Configuration` tab. On the `Configuration` page you would configure appropriately, its all straight forwards, but, take your time and remember to save settings, and then load model. If the model loads correctly it will say so in the `Status Bar` on the bottom od the display. On Ubuntu if your default sound card is not in the list, then you should ensure to set the Sound Card.
3. Go back to the `Interaction` page and begin interactions, ensuring to notice tool options available, and select appropriately for your intended use, then type your input into the User Input box, and then click Send Input.
4. When all is finished, click `Exit` on the bottom right, then you are left with the terminal menu, where you type `x` to exit.
```

### Useful Info
- Research Tools, configuration scales to Context Length...
```
DDG Search = Faster DuckDuckGo research.
Web Search = Slower actual website reading research. 

| Context | Multiplier | DDG Results | DDG Deep | Web Results | Web Deep |
|---------|------------|-------------|----------|-------------|----------|
| 16384   | 0.5x       | 4           | 2        | 6           | 3        |
| 32768   | 1.0x       | 8           | 4        | 12          | 6        |
| 65536   | 2.0x       | 16          | 8        | 24          | 12       |
| 131072  | 4.0x       | 32          | 16       | 48          | 24       |
```
- The Large Embedding Model requires SIGNIFICANTLY more RAM...
```
| Specification          | Small (bge-small-en-v1.5) | Medium/Base (bge-base-en-v1.5) | Large (bge-large-en-v1.5) |
| ---------------------- | ------------------------- | ------------------------------ | ------------------------- |
| **Parameters**         | ~33M (33 million)         | ~109M (109 million)            | ~335M (335 million)       |
| **Disk Size**          | ~130 MB                   | ~420 MB                        | ~1.3 GB (1,300 MB)        |
| **RAM Required**       | 300–500 MB                | 1.0–1.5 GB                     | 3.0–4.5 GB\*              |
| **Avg Speed**          | ~2,000 sent/sec           | ~800 sent/sec                  | ~150 sent/sec             |
| **MTEB Score**\*       | ~62.0                     | ~63.5                          | ~64.5                     |
```

### Notation 
- Because of how the WT-Web interface works, there is ongoing issue with how sometimes lists of values in the configuration page are non-selectable; just select a different number in list first, then select number you want. Only way to fix this would be to drop older windows/ubuntu support.  
- The "Cancel Input/Response" button was impossible for now; Attempted most recently, 2 Opus 4.5 and 2 Grok, sessions, and added about ~=>45k, but then required, "Wait For Response" for Gradio v3 and Cancel Input for Gradio v4. Instead there is now a dummy "..Wait For Response.." button.
- Support was maintained for Windows 7-8; FastEmbed/ONNX was replaced with PyQt5 + Qt5 WebEngine. So its slower, but the plan is Windows 7-11 and Ubuntu 22-25. Other optimized projects may follow.
- Optimize context length; the chatbot will chunk data to the size of the context length, however using a max_context_length of ~128000 is EXTREMELY SLOW, and on older computers try NOT to use a context_length over ~32000. 
- The "iMatrix" models do not currently work, due to requirement of Cuda for imatrix to work. Just to save some issues for people that dont know.
- VRAM dropdown, 1GB to 64GB in steps, this should be your FREE ram available on the selected card, if you are using the card at the time then this is why we have for example, 6GB for a 8GB GPU in use, safely allowing 2GB for the system, while in compute more one would use for example, the full 8GB on the 8GB GPU.
- I advise GPU can cover the Q6_K version, the Q6_K useually has negligable quality loss, while allowing good estimation of if the model will fit on a card, ie 8GB card will typically be mostly/completely cover a 7B/8B model in Q6_K compression, with a little extra to display the gradio interface, so the numbers somewhat relate with Q6_K when using same card as display.
- We use a `1.125` additional to model size for layers to fit on VRAM,  the calculation is `TotalVRam /((ModelFileSize * 1.125) / NumLayers = LayerSize) = NumLayersOnGpu`.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.
- "Chat-Windows-Gguf" and "Chat-Linux-Gguf", is now "Chat-Gradio-Gguf", as yes, these dual-mode scripts used to be 2 different/same programs.
- Through detection/use of flags AVX/AVX2/AVX512, FMA, F16C, then supposedly we can expect ≈ 1.4 – 1.6× the tokens-per-second you would get from a plain AVX2-only build and roughly half the RAM footprint when you load FP16-quantised GGUF files.

### Models working (with gpt for comparrisson). 
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) (20B)      | 84.1%  | 58.1%   | 96.0-98.7%   | 71.5%    | ~42.5%    | 85.3%           | N/A   |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B-A3B)      | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |
| [Lamarckvergence-14B-GGUF](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) (14B)          | 76.56%   | 50.33%   | 54.00%   | 15.10%   | 16.34%   | 47.59% (MMLU-PRO) | N/A       |
| qwen2.5-test-32b-it (32B)              | 78.89%   | 58.28%   | 59.74%   | 15.21%   | 19.13%   | 52.95%            | 29.54 kg  |
| [T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF/tree/main) (14B)      | ~73.24%   | ~65.47%   | ~28.63%   | ~22.26%    | ~38.69%   | ~54.27% (MMLU-PRO) | ~1.56 kg   |
| [Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF) (7B)     | ~76.40%   | ~36.62%   | ~48.79%   | ~8.95%    | ~15.51%   | ~37.51% (MMLU-PRO) | ~0.62 kg   |
| [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) (7B)          | 75.83%   | 36.36%   | 48.39%   | 7.72%    | 15.17%   | 37.73% (MMLU-PRO) | N/A       |
| [Q2.5-R1-3B-GGUF](https://huggingface.co/mradermacher/Q2.5-R1-3B-GGUF) (3B)                    | 42.14%   | 27.20%   | 26.74%   | 7.94%    | 12.73%   | 31.26% (MMLU-PRO) | N/A       |

<details>
  <summary>Table Key ></summary>

    - IFEval (Instruction-Following Evaluation) - Measures how well an AI model understands and follows natural language instructions.
    - BBH (Big-Bench Hard) - A challenging benchmark testing advanced reasoning and language skills with difficult tasks.
    - MATH - Evaluates an AI model’s ability to solve mathematical problems, from basic to advanced levels.
    - GPQA (Graduate-Level Google-Proof Q&A) - Tests an AI’s ability to answer tough, graduate-level questions that require deep reasoning, not just web lookups.
    - MuSR (Multi-Step Reasoning) - Assesses an AI’s capability to handle tasks needing multiple logical or reasoning steps.
    - MMLU (Massive Multitask Language Understanding) - A broad test of general knowledge and understanding across 57 subjects, like STEM and humanities.
    - CO2 Cost - Quantifies the carbon dioxide emissions from training or running an AI model, reflecting its environmental impact.

</details>

## Structure
- Core Project files...
```
project_root/
│ Chat-Windows-Gguf.bat
│ installer.py
│ validater.py
│ launcher.py
├── media/
│ └── project_banner.jpg
├── scripts/
│ └── browser.py
│ └── display.py
│ └── inference.py
│ └── configuration.py
│ └── tools.py
│ └── utlity.py
```
- Installed/Temporary files...
```
project_root/
├── data/
│ └── persistent.json
│ └── constants.ini
├── data/vectors/
└─────── *
├── data/temp/
└────── *
├── data/history
└────── *
├── .venv/
└────── *
```

- Built in Tts tool notes.
```
| Mode     | TTS engine       | Audio *player* |
| -------- | ---------------- | -------------- |
| Windows  | `pyttsx3` (SAPI) | **built-in**   |
| Pulse    | `espeak-ng`      | `paplay`       |
| PipeWire | `espeak-ng`      | `pw-play`      |
```

# Development for v1.00.00+
Now that the program is 100% made, I am tidying up, and finishing testing/bugfixing... 
- Optimization, then testing/bugfixing.
- Rename configuration.py to configure.py or config.py.
- More work on filtering output. Specifically enhancing the "Light" filter for Qt6, as currently its got too many blank lines.
- Build options on Ubuntu 3/4 on first menu, are currently broken. Download install through 1/2 on first menu work ok, and all options work fine on Windows, but needs to be fixed for Ubuntu.
- Issues with Large Embedding model, I dont advise use of Large model anyhow, it needs, 1.5GB file size but main thing 4GB System RAM, with CPU inference and consistently either is, SLOWER or CRASHING. The code was assessed and updated with Kimi K2.5, yet issues remain, it works mostly on Windows, but can still fail and is unreliable.

# Update Ideas
1. Thinking visualisation. The visualization of nodes and intersecting lines would be embedded in teh log, or what would be the best way to do this? for example instead of current think phase visualization.
2. Possibility of artifacts, the AI mentioned this, I want to see how feasable it is first.
4. Image reading (this would additionally require vllm, which could switch for such iterations involving image reading).
5. new project Chat-Lightwave-Gguf, will be optimized towards windows 8-10, but use MS Edge WebView, and use lightwave instead of gradio, and limit operation to Windows 8.1-11.

## Credits
Thanks to all the following teams, for the use of their software/platforms...
- [Llama.Cpp](https://github.com/ggml-org/llama.cpp) - The binaries used for interference with models.
- [Claude Sonnet/Opus](https://claude.ai/chat), [Kimi K2/K2.5](https://www.kimi.com), [XAI Grok](https://x.com/i/grok), [Deepseek R1/3](https://www.deepseek.com/), [Perplexity](https://www.perplexity.ai) - Paid/Free AI Platforms. 
- Python Libraries - More main libraries used need listing here.

## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, if you wish to use most of the code, then you should fork, or, if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead, but, do not claim, my vanilla or =>50% of my, releases as your own work`.

