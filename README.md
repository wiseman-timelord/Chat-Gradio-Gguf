# ![Chat-Windows-Gguf](media/project_banner.jpg)
Status - v1 Beta / v2 Alpha
- v2 (upcoming), moving on to version 2, where I will limit support to, Python 3.11-3.13 and Windows 10-11 and Ubuntu 24-25, in order to get, newer libraries and slightly more manageable code complexity. The readme.md here will be tailored towards the v2 release from now.
- v1, after ~v1.10.x I tried to implement current model handling, ie qwen3.5 deepseek 3.2(?), etc, and clearly claude or whatever else I used was confused, and corrupted things. For now you would have to fish around in the releases for a version that runs for example Qwen3 level models. I will release a final v1 later, and fully check things. v1 is, Windows 7-11 and ubuntu 22-25, compatible. 

## Description
Intended as a high-quality chat interface with wide hardware/os support, windows 10-11 (WSL not required) and Ubuntu 24-25, with any Gpu on GGUF models through Python ~3.11-~3.13. An optimal number of features for a ChatBot, as well as, dynamic buttons/panels on the interface and websearch and RAG and TTS and archiving of sessions, and all on local models, so no imposed, limitations or guidelines (model dependent). This tool providing a comparable interface found on premium non-agentic AI services, where the configuration is intended to be intelligent, while without options reported in forums to make no difference on most models (no over-complication). The program using offline libraries (apart from websearch) instead of, online services or repeat download or registration.

### Core Principles
- This project is for a chat interface, and is not intended to overlap with other blueprints/projects, `Rpg-Gradio-Gguf` or `Agent-Gradio-Gguf`. 
- A, Windows 10-11 and Ubuntu 24-25, compatibility range; Any and all compatibility issues within those ranges MUST be overcome.
- This Program is also intended to have only basic tools of, DDG Search, Web Search, and Speech Out. Though more may be added later.
- I am making this program for the community, so that people on Ubuntu and Windows, are able to use a single chatbot for AI, on recent models.

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
- The Y1400 configuration (requiring restart) on a Y1920 Portrait mode display on Windows, with both side panels collapsed, optimally displaying text. (using the model Qwen3.5-4B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled-GGUF) (v1.11.0)...
![image_missing](media/expanded_portrait.jpg)

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

- Startup looks like this in the Command Prompt console (v1.10.6)...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf: Launcher
    ===============================================================================
    
    Starting Chat-Gradio-Gguf...
    [COMPAT] NLP uses word-split fallback
    [COMPAT] sentence_transformers loaded
    [COMPAT] Pydantic v1.10.21 — Gradio 3.x compatible
    `main` Function Started.
    [INI] Platform: windows
    [INI] Backend: VULKAN_CPU
    [INI] Vulkan: True
    [INI] Graphics Acceleration: True
    [INI] Qt Version: 5 (v5)
    [INI] DX Feature Level: 0xb100
    [INI] Embedding Model: BAAI/bge-small-en-v1.5
    [INI] Gradio Version: 3.50.2
    [INI] OS Version: 10
    [INI] Windows Version: 10
    [INI] TTS Type: Built-in (pyttsx3/espeak-ng)
    [MODELS] Scanning directory: G:\LargeModels\Size_ittle_1b-2b
    [MODELS] ✓ Found 9 models:
    [MODELS]   - Benchmaxx-Llama-3.2-1B-Instruct.Q6_K.gguf
    [MODELS]   - DeepSeek-V3-1B-Test-Q6_K.gguf
    [MODELS]   - deepseek-v3-tiny-random.Q6_K.gguf
    [MODELS]   - Dolphin3.0-Qwen2.5-0.5B-Q4_K_M.gguf
    [MODELS]   - gemma-3-1b-thinking-v2-q6_k.gguf
    [MODELS]   ... and 4 more
    [CONFIG] Loaded -> Model: qwen1_5-0_5b-chat-q3_k_m.gguf | CPU: Auto-Select
    Configuration loaded
    [TTS] Engine: pyttsx3
    [TTS] Audio Backend: windows
    [TTS] Voice from config: Microsoft Zira  - English (United States) (en-US) (HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0)
    [Vulkan] GGML_CUDA_NO_PINNED=1   (frees ~300 MB VRAM)
    [Vulkan] GGML_VK_NO_PIPELINE_CACHE=0  (cached SPIR-V pipelines)
    Script mode `windows` with backend `VULKAN_CPU`
    Working directory: ...les\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.10.5
    Data Directory: ...hat-Gradio-Gguf\Chat-Gradio-Gguf-1.10.5\data
    Session History: ...io-Gguf\Chat-Gradio-Gguf-1.10.5\data\history
    Temp Directory: ...radio-Gguf\Chat-Gradio-Gguf-1.10.5\data\temp
    [CPU] Detected: 12 cores, 24 threads
    [CPU] Current: 20
    CPU Configuration: 12 physical cores, 24 logical cores
    
    Configuration:
      Backend: VULKAN_CPU
      Model: qwen1_5-0_5b-chat-q3_k_m.gguf
      Context Size: 8192
      VRAM Allocation: 8192 MB
      CPU Threads: 20
      GPU Layers: 0
    
    [INIT] Pre-loading auxiliary inference...
    [INIT] WARN spaCy model not available (will use fallback)
    [RAG] Loading embedding model: BAAI/bge-small-en-v1.5
    [RAG] Downloading/loading to: C:\Inference_Files\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.10.5\data\embedding_cache
    Loading weights: 100%|███████████████████████████████████████████| 199/199 [00:00<00:00, 11004.75it/s]
    BertModel LOAD REPORT from: BAAI/bge-small-en-v1.5
    Key                     | Status     |  |
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  |
    
    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    [RAG] Embedding model loaded successfully (dim=384)
    [INIT] OK Embedding model pre-loaded from cache
    
    Launching Gradio display...
    [DISPLAY] Qt Version: 5 (v5)
    [DISPLAY] Gradio Version: 3.50.2
    [DISPLAY] Graphics Acceleration: True
    [FILTER] Using gradio3 filter (15 rules)
    IMPORTANT: You are using gradio version 3.50.2, however version 4.44.1 is available, please upgrade.
    --------
    [MODELS] Scanning directory: G:\LargeModels\Size_ittle_1b-2b
    [MODELS] ✓ Found 9 models:
    [MODELS]   - Benchmaxx-Llama-3.2-1B-Instruct.Q6_K.gguf
    [MODELS]   - DeepSeek-V3-1B-Test-Q6_K.gguf
    [MODELS]   - deepseek-v3-tiny-random.Q6_K.gguf
    [MODELS]   - Dolphin3.0-Qwen2.5-0.5B-Q4_K_M.gguf
    [MODELS]   - gemma-3-1b-thinking-v2-q6_k.gguf
    [MODELS]   ... and 4 more
    [BROWSER] Starting Gradio server in background...
    [BROWSER] Waiting for Gradio server at http://localhost:7860...
    Running on local URL:  http://localhost:7860
    
</details>

- The combined Info/Install menu  (v2.0.0)...
<details>
    
    ===============================================================================
     Chat-Gradio-Gguf v2 — Backend & Install Size
    ===============================================================================
    
    System Detections...
       CPU Features : AVX | AVX2 | FMA | F16C
       Build Tools  : Git OK | CMake OK | MSVC OK | MSBuild OK
       Platform     : Windows 10 | Python 3.12
       GPU          : DX11.1 | Vulkan: YES
    
    Backend Options...
       1) Download CPU Binary / Default CPU Wheel (Wheel v0.3.16)
       2) Download Vulkan Binary / Default CPU Wheel (Wheel v0.3.16)
       3) Compile CPU Binaries / Compile CPU Wheel (Wheel v0.3.22)
       4) Compile Vulkan Binaries / Compile Vulkan Wheel (Wheel v0.3.22)
    
    Install Size...
       a) Small  +450MB  - Bge-Small-En v1.5 + Coqui TTS (faster)
       b) Medium +1.5GB  - Bge-Base-En v1.5  + Coqui TTS (quality)
    
    ===============================================================================
    Selection; Backend=1-4, Size=a-b, Abandon=A; (e.g. 2b):

</details>

- The installation without compiling...(v2.00.0)
<details>
    
    ===============================================================================
     Chat-Gradio-Gguf v2 — Installation
    ===============================================================================
    
    Installing Chat-Gradio-Gguf v2 on Windows 10 with Python 3.12
      Mode: Clean Install
      Route: Download Vulkan Binary / Default CPU Wheel
      Llama.Cpp b8882, Gradio 5.x, Qt-Web v6
      Embedding: BAAI/bge-small-en-v1.5
      GPU: DirectX Feature Level 0xb100
      TTS: Coqui (p225,p226 / english)
    [✓] Removed existing virtual environment
    [✓] Created new virtual environment
    [✓] Upgraded pip to latest version
    [✓] Verified virtual environment setup
    [✓] Installing py-cpuinfo for CPU detection...
    Collecting py-cpuinfo
      Using cached py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
    Using cached py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
    Installing collected packages: py-cpuinfo
    Successfully installed py-cpuinfo-9.0.0
    
    [notice] A new release of pip is available: 24.0 -> 26.1
    [notice] To update, run: C:\Inference_Files\Chat-Gradio-Gguf\Chat-Gradio-Gguf-1.10.9\.venv\Scripts\python.exe -m pip install --upgrade pip
    [✓] py-cpuinfo installed
    [✓] Protected directory preserved: data/embedding_cache
    [✓] Directories created/verified
    [✓] System information file created
    [✓] Installing Python dependencies...
    [✓] Installing packages (Gradio 5.x)...
      [1/20] Installing numpy...      Installing collected packages: numpy
     OK
      [2/20] Installing requests...      Installing collected packages: urllib3, idna, charset_normalizer, certifi, requests
     OK
      [3/20] Installing pyperclip...      Installing collected packages: pyperclip
     OK
      [4/20] Installing spacy...      Installing collected packages: wrapt, typing-extensions, spacy-loggers, spacy-legacy, shellingham, setuptools, pygments, packaging, murmurhash, mdurl, MarkupSafe, h11, cymem, confection, colorama, cloudpathlib, catalogue, blis, annotated-types, annotated-doc, wasabi, typing-inspection, tqdm, srsly, smart-open, pydantic-core, preshed, markdown-it-py, jinja2, httpcore, click, anyio, rich, pydantic, httpx, typer, thinc, weasel, spacy
     OK
      [5/20] Installing psutil...      Installing collected packages: psutil
     OK
      [6/20] Installing ddgs...      Installing collected packages: primp, lxml, ddgs
     OK
      [7/20] Installing langchain-community...      Installing collected packages: zstandard, xxhash, uuid-utils, tenacity, PyYAML, python-dotenv, propcache, orjson, mypy-extensions, multidict, marshmallow, langchain-protocol, jsonpointer, httpx-sse, greenlet, frozenlist, attrs, aiohappyeyeballs, yarl, typing-inspect, SQLAlchemy, requests-toolbelt, jsonpatch, aiosignal, pydantic-settings, langsmith, dataclasses-json, aiohttp, langchain-core, langchain-text-splitters, langchain-classic, langchain-community
     OK
      [8/20] Installing langchain-text-splitters...   OK
      [9/20] Installing faiss-cpu...      Installing collected packages: faiss-cpu
     OK
      [10/20] Installing langchain...      Installing collected packages: ormsgpack, langgraph-sdk, langgraph-checkpoint, langgraph-prebuilt, langgraph, langchain
     OK
      [11/20] Installing pygments...   OK
      [12/20] Installing lxml...      Installing collected packages: lxml
            Uninstalling lxml-6.1.0:
     OK
      [13/20] Installing lxml_html_clean...      Installing collected packages: lxml_html_clean
     OK
      [14/20] Installing beautifulsoup4...      Installing collected packages: soupsieve, beautifulsoup4
     OK
      [15/20] Installing aiohttp...   OK
      [16/20] Installing newspaper4k...      Installing collected packages: sgmllib3k, brotli, w3lib, six, pillow, filelock, feedparser, requests-file, python-dateutil, tldextract, newspaper4k
     OK
      [17/20] Installing pywin32...      Installing collected packages: pywin32
     OK
      [18/20] Installing tk...      Installing collected packages: tk
     OK
      [19/20] Installing pythonnet...      Installing collected packages: pycparser, cffi, clr_loader, pythonnet
     OK
      [20/20] Installing gradio...        Downloading pytz-2026.2-py2.py3-none-any.whl.metadata (22 kB)
        Downloading pytz-2026.2-py2.py3-none-any.whl (510 kB)
        Installing collected packages: pytz, pydub, websockets, tzdata, tomlkit, semantic-version, ruff, python-multipart, pydantic-core, pillow, hf-xet, groovy, fsspec, ffmpy, aiofiles, uvicorn, starlette, pydantic, pandas, safehttpx, fastapi, huggingface-hub, gradio-client, gradio
            Uninstalling pydantic_core-2.46.3:
            Uninstalling pillow-12.2.0:
            Uninstalling pydantic-2.13.3:
     OK
    [✓] Base packages installed
    [✓] Installing PyTorch (CPU) — torch>=2.5.0...
          Using cached https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl.metadata (29 kB)
        Using cached https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl (114.5 MB)
        Installing collected packages: mpmath, sympy, setuptools, networkx, torch
            Uninstalling setuptools-82.0.1:
    [✓] PyTorch (CPU) installed
    [✓] setuptools restored after torch install
    [✓] Installing transformers>=4.44.0...
        Installing collected packages: safetensors, regex, tokenizers, transformers
    [✓] transformers installed
    [✓] Installing sentence-transformers>=3.3.0...
        Installing collected packages: threadpoolctl, setuptools, scipy, joblib, scikit-learn, sentence-transformers
            Uninstalling setuptools-82.0.1:
    [✓] sentence-transformers installed
    [✓] Embedding backend verified
    [✓] Installing Qt6 WebEngine for custom browser...
        Installing collected packages: PyQt6-Qt6, PyQt6-sip, PyQt6
        Installing collected packages: PyQt6-WebEngine-Qt6, PyQt6-WebEngine
    [✓] Qt6 WebEngine installed successfully
    [✓] Installing llama-cpp-python 0.3.16 (CPU, trying 3 sources)...
      Trying: eswarthammana/llama-cpp-wheels 0.3.16
          Downloading https://github.com/eswarthammana/llama-cpp-wheels/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl (4.2 MB)
        Installing collected packages: diskcache, llama-cpp-python
    [✓] llama-cpp-python 0.3.16 installed via eswarthammana/llama-cpp-wheels 0.3.16
    [✓] Python dependencies installed successfully
    [✓] Installing optional file format support...
    [✓]   Installed PyPDF2
    [✓]   Installed python-docx
    [✓]   Installed openpyxl
    [✓]   Installed python-pptx
    [✓] Initializing embedding cache for BAAI/bge-small-en-v1.5...
    Embedding Initialization Output...
        Importing torch...
        torch version: 2.11.0+cpu
        CUDA available: False (should be False)
        Importing sentence_transformers...
        Loading model: BAAI/bge-small-en-v1.5
        Testing embedding...
        SUCCESS: Model loaded, dimension: 384
    [✓] Embedding cache initialized
    [✓] Downloading spaCy language model...
    Downloading spaCy model: [==============================] 100% (12.2MB/12.2MB) - Complete
    [✓] Installing spaCy model...
    [✓] spaCy model installed
    [✓] Installing espeak-ng (Coqui dependency)...
    [✓] espeak-ng verified
    [✓] Installing Coqui TTS with codec support...
    [✓] Installing torchaudio (CPU-only to match torch)...
          Using cached https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl.metadata (7.0 kB)
        Using cached https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-win_amd64.whl (326 kB)
        Installing collected packages: torchaudio
    [✓] torchaudio (CPU) installed
    [✓] Coqui TTS package installed
    [✓] Patched autoregressive.py for transformers compatibility
    [✓] Downloading Coqui VCTK voice model (~1.4GB)...
    [COQUI] espeak-ng directory added to PATH
    [COQUI] espeak-ng DLL: C:/Inference_Files/Chat-Gradio-Gguf/Chat-Gradio-Gguf-1.10.9/data/espeak-ng\libespeak-ng.dll
    [COQUI] espeak-ng data: C:/Inference_Files/Chat-Gradio-Gguf/Chat-Gradio-Gguf-1.10.9/data/espeak-ng\espeak-ng-data
    [COQUI] espeak-ng verified
    [COQUI] Loading model...
    [COQUI] Testing synthesis...
    [COQUI] Model test passed
    [✓] Coqui TTS installed and verified
    [✓] Downloading backend binaries...
    Downloading backend: [==============================] 100% (32.4MB/32.4MB) - Complete
    [✓] Extracting backend...
    Extracting: [=========================] 100% (43.0B/43.0B)
    [✓] Backend ready
    [✓] Configuration file created
    [✓] Installation complete!
    
    Run the launcher to start Chat-Gradio-Gguf v2
    
    [✓] Cleaned up compilation temp folder
    Press any key to continue . . .

</details>

## Hard Requirements
- Windows 10-11 and/or ~Ubuntu 24-25 - Its BOTH a, Windows AND linux, program, batch for windows and bash for linux, launch dual-mode scripts.
- Python ~3.11-~3.13 - Requires [Python](https://www.python.org); AI warns me certain libraries wont work on Python 3.14, possibly Spacy. 
- Llm Model - You will need a Large Language Model in GGUF format, check the models section for recommendations, but for quick start I advise one like [Qwen3-4B-abliterated-GGUF](https://huggingface.co/mradermacher/Qwen3-4B-abliterated-GGUF) for testing basic operation.
- Suitable GPU - Gpu may be, Main or Compute, with VRam selection 4GB-96GB. Ideally you want the GPU to cover all model layers for fast interference.
- System Ram - Your system ram must cover, the curren load and the size of the model layers not able to be covered by the GPU, plus smaller models like the embeddings model, plus the wheel if not built for Vulkan.

### Building Requirements 
For compile options; If on PATH, ask AI how to check its on path, and as applicable fix...
- [MSVC++ 2017-2022](https://visualstudio.microsoft.com/vs/older-downloads/) - MSVC with option Desktop Development enabled during install.
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
- Windows is my main OS, so Check for recent v1 versions mentioning work on ubuntu if you use that (it was likely being tested/used at such points).
- Changing height of session log height as shown in media section requires restart of GUI, to re-initialize the complicated and fragile dynamic UI through browser fake GUI interface. 
- The "Cancel Input/Response" button was impossible for now; Attempted most recently, 2 Opus 4.5 and 2 Grok, sessions, and added about ~=>45k, but then required, "Wait For Response" for Gradio v3 and Cancel Input for Gradio v4. Instead there is now a dummy "..Wait For Response.." button.
- Optimize context length; the chatbot will chunk data to the size of the context length, however using a max_context_length of ~128000 is EXTREMELY SLOW, and on older computers try NOT to use a context_length over ~32000. 
- The "iMatrix" models do not currently work, due to requirement of Cuda for imatrix to work. Just to save some issues for people that dont know. In other words, if the model has "i" in its label somewhere significant, then likely it is a iMatrix model, and you will need some other ChatBot that handles such things.
- VRAM dropdown, 1GB to 96GB in steps, this should be your FREE ram available on the selected card, if you are using the card at the time then this is why we have for example, 6GB for a 8GB GPU in use, safely allowing 2GB for the system, while in compute more one would use for example, the full 8GB on the 8GB GPU.
- I advise GPU can cover the Q5_KM/6_K versions of models, these useually has negligable quality loss, while allowing good estimation of if the model will fit on a card, ie 8GB card will typically be mostly/completely cover a 7B/8B model in 5_KM/Q6_K compression, with a little extra to display the gradio interface, so the numbers somewhat relate with Q5_KM/Q6_K when using same card as display.
- We used a `1.125` additional to model size for layers to fit on VRAM,  the calculation is `TotalVRam /((ModelFileSize * 1.125) / NumLayers = LayerSize) = NumLayersOnGpu`. This possibly is not the case now.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.
- "Chat-Windows-Gguf" and "Chat-Linux-Gguf", is now "Chat-Gradio-Gguf", as yes, these dual-mode scripts used to be 2 different/same programs.
- Through detection/use of flags AVX/AVX2/AVX512, FMA, F16C, then supposedly we can expect ≈ 1.4 – 1.6× the tokens-per-second you would get from a plain AVX2-only build and roughly half the RAM footprint when you load FP16-quantised GGUF files.

### Models working (with gpt for comparrisson).
- Models newer than Qwen 3 will (at the current time/date) require to compile during install, this is due to a versioning difference between, the [available pre-built wheels](https://github.com/eswarthammana/llama-cpp-wheels/releases) and the [latest llama.cpp binary](https://github.com/ggml-org/llama.cpp/releases). Hope that makes sense. Either way, the models just keep getting better...

| Model                                  | IFEval   | BBH      | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [Qwen3.5-9B](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF) | 91.5%    | N/A      | N/A      | 81.7% (Diamond) | N/A      | 82.5% (MMLU-Pro)  | N/A       |
| [Qwen3.5-27B](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF) | 95.0%    | N/A      | N/A      | 85.5% (Diamond) | N/A      | 86.1% (MMLU-Pro)  | N/A       |
| [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) (20B) | 84.1%  | 58.1%   | 96.0-98.7%   | 71.5%    | ~42.5%    | 85.3%           | N/A   |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B-A3B) | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |
| **Qwen3-8B**       | ~83%     | 56.73%   | 60.80%   | 44.44%   | N/A      | 76.89%            | N/A       |
| [Lamarckvergence-14B-GGUF](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) (14B) | 76.56%   | 50.33%   | 54.00%   | 15.10%   | 16.34%   | 47.59% (MMLU-PRO) | N/A       |
| qwen2.5-test-32b-it (32B)              | 78.89%   | 58.28%   | 59.74%   | 15.21%   | 19.13%   | 52.95%            | 29.54 kg  |
| [T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF/tree/main) (14B) | ~73.24%   | ~65.47%   | ~28.63%   | ~22.26%    | ~38.69%   | ~54.27% (MMLU-PRO) | ~1.56 kg   |
| [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) (7B) | 75.83%   | 36.36%   | 48.39%   | 7.72%    | 15.17%   | 37.73% (MMLU-PRO) | N/A       |


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
│ launcher.py
├── media/
│ └── project_banner.jpg
├── scripts/
│ └── configure.py
│ └── display.py
│ └── inference.py
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
├── scripts/
│ └── __init__.py
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

# Development for next release
More testing was done, the restults were not good, moved on to v2. There has been an overhaul, its it seems to work correctly, the next plans are...
1. v2 - Limit support to, Python 3.11-3.13 and Windows 10-11 and Ubuntu 24-25, in order to get, newer libraries and slightly more manageable code complexity. all features to then be tested, and multiple models tested, likely the model handling will inherit the issues from before, but be easier to fix. 
2. v1 - Need to clean up v1, and do full test, including Testing/Bugfixing on Ubuntu 22 as this also needs to be checked/done. The issue is gradio becoming a set version 3, requring a shim for pydantic to work correctly. Problems started around there, so maybe we can cut whatever pydantic does from v1. Idea being to make a final version to leave it, after which later only updating to sync model handling.
3. When there is a stable correctly working latest v2, then the plan will be to develop a stack of tools, ie "Web Research", but this needs to be brainstormed.
4. Need to ensure attachments is working correctly, and if/when so, then we need artifacts, ie the ability to reference files provided earlier in the session and their presentation.

# Update Ideas
- It would be great to have a Auto/Quick/Think toggle, so as to have some pre-configured 1b model or the likes be able to do a quick assessment of, if the request is simple or complete, to then, disable think for simple tasks and enable think for complex tasks, and additionally we would want to have it assess context/batch lengths, if the user additionally has those set to auto. Other settings could also be determined through this auto feature pre-prompt system. Needs brainstorming.
- Code Optimization / redistribution of code.
- Thinking visualisation. The visualization of nodes and intersecting lines would be embedded in teh log, or what would be the best way to do this? for example instead of current think phase visualization.
- Possibility of artifacts, the AI mentioned this, I want to see how feasable it is first.
- Image reading (this would additionally require vllm, which could switch for such iterations involving image reading).

## Credits
Thanks to all the following teams, for the use of their software/platforms...
- [Llama.Cpp](https://github.com/ggml-org/llama.cpp) - The binaries used for interference with models.
- [Claude Sonnet/Opus](https://claude.ai/chat), [Kimi K2/K2.5](https://www.kimi.com), [XAI Grok](https://x.com/i/grok), [Deepseek R1/3](https://www.deepseek.com/), [Perplexity](https://www.perplexity.ai) - Paid/Free AI Platforms. 
- Python Libraries - More main libraries used need listing here.

## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, if you wish to use most of the code, then you should fork, or, if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead, but, do not claim, my vanilla or =>50% of my, releases as your own work`.

