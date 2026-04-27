# ![Chat-Windows-Gguf](media/project_banner.jpg)

### Status
Beta - Latest release v1.10.9 was completion of overhaul, and tested working with Qwen 3.5/3.6 on windows 10, however it turn out other models are now not working, so I'd wait for next release for version that can run both newer and older models, as the newer models are the focus. Testing/bugfixing on Ubuntu needs to be done, though past releases mention ubuntu work phases.

## Description
Intended as a high-quality chat interface with wide hardware/os support, windows 7-11 (WSL not required) and Ubuntu 22-25, with any Gpu on GGUF models through Python ~3.9-3.13. An optimal number of features for a ChatBot, as well as, dynamic buttons/panels on the interface and websearch and RAG and TTS and archiving of sessions, and all on local models, so no imposed, limitations or guidelines (model dependent). This tool providing a comparable interface found on premium non-agentic AI services, where the configuration is intended to be intelligent, while without options reported in forums to make no difference on most models (no over-complication). The program using offline libraries (apart from websearch) instead of, online services or repeat download or registration. One thing to mention though, is that because a goal of the project is to make a Chatbot compatible as it is, therefore compromises had to be made, and it may not have all the latest versions of things, more the most widely compatible versions of things, so some features are not as good as they could be, but also, this program serves as a valuable reference for what needs to be used on what OS, albeit it, as a personal project it could be streamlined/enhanced/customized to your own likings later.   

### Core Principles
- This project is for a chat interface, and is not intended to overlap with other blueprints/projects, `Rpg-Gradio-Gguf` or `Agent-Gradio-Gguf`. 
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
- The Interaction page showing the Web Search tool. Windows using the model Qwen3 4B Abliterated (v1.10.8)...
![image_missing](media/conversation_page_windows.jpg)

- The Y1400 Portrait display configuration optimally displaying text. Windows using the model Qwen3.5-4B-Abliterated-Claude-4.6-Opus-Reasoning-Distilled-GGUF (v1.11.0)...
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

- The combined Info/Install menu  (v1.10.3)...
<details>
    
    =================================================================================
        Chat-Gradio-Gguf - Configure Installation
    =================================================================================
    
    System Detections...
        CPU Features: SSE3, SSSE3, SSE4_1
        Operating System: Windows 10
        Optimal Python: 3.12; Vulkan Present: No
        Build Tools: Git, CMake, MSVC, MSBuild
    
    Backend Options...
       1) Download CPU Binary / Default CPU Wheel
       2) Download Vulkan Binary / Default CPU Wheel
       3) Compile CPU Binaries / Compile CPU Wheel
       4) Compile Vulkan Binaries / Compile Vulkan Wheel
    
    Install Size...
       a) Small  ~135MB - Bge-Small-En v1.5 + pyttsx3/espeak-ng (built-in TTS)
       b) Medium ~450MB - Bge-Base-En v1.5 + pyttsx3/espeak-ng (built-in TTS)
       c) Large   ~2GB  - Bge-Base-En v1.5 + Coqui TTS (high quality voices)
    
    =================================================================================
    Selection; Backend=1-4, Size=a-c, Abandon=A; (e.g. 2b):

</details>

- The installation with compiling...(v1.04.2)
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf - Installation
    ===============================================================================
    
    Installing Chat-Gradio-Gguf on Windows 10 with Python 3.12
      Route: Compile Vulkan Binaries / Compile Vulkan Wheel
      Llama.Cpp b7688, Gradio 5.49.1, Qt-Web v6
      Embedding: BAAI/bge-base-en-v1.5
      TTS: Coqui - British (Male/Female)
    [✓] Verified directory: data
    [✓] Verified directory: scripts
    [✓] Verified directory: models
    [✓] Verified directory: data/history
    [✓] Verified directory: data/temp
    [✓] Verified directory: data/vectors
    [✓] Verified directory: data/embedding_cache
    [✓] Using build temp path: C:\temp_build
    [✓] System information file created
    [✓] Installing Python dependencies...
    [✓] Installing base packages (Gradio 5.49.1)...
      [1/21] Installing numpy<2... OK
      [2/21] Installing requests... OK
      [3/21] Installing pyperclip... OK
      [4/21] Installing spacy... OK
      [5/21] Installing psutil... OK
      [6/21] Installing ddgs... OK
      [7/21] Installing langchain-community... OK
      [8/21] Installing faiss-cpu... OK
      [9/21] Installing langchain... OK
      [10/21] Installing pygments... OK
      [11/21] Installing lxml... OK
      [12/21] Installing lxml_html_clean... OK
      [13/21] Installing tokenizers... OK
      [14/21] Installing beautifulsoup4... OK
      [15/21] Installing aiohttp... OK
      [16/21] Installing pywin32... OK
      [17/21] Installing tk... OK
      [18/21] Installing pythonnet... OK
      [19/21] Installing pyttsx3... OK
      [20/21] Installing newspaper4k... OK
      [21/21] Installing gradio... OK
    [✓] Base packages installed
    [✓] Installing embedding backend (torch + sentence-transformers)...
    [✓] Installing PyTorch 2.4+ (CPU-only) for Python 3.12...
    [✓] PyTorch 2.4+ (CPU) installed
    [✓] Installing transformers>=4.42.0...
    [✓] transformers installed
    [✓] Installing sentence-transformers>=3.0.0...
    [✓] sentence-transformers installed
    [✓] Embedding backend verified
    [✓] Installing Qt WebEngine for custom browser...
    [✓] Windows 10 - installing PyQt6 + Qt6 WebEngine...
    [✓] Qt6 WebEngine installed
    [✓] Vulkan wheel build - checking Vulkan SDK...
    [✓] Vulkan SDK detected
    [✓] Building llama-cpp-python from source (10-20 minutes)
      Using 20 parallel build threads
      Build flags: GGML_VULKAN=1, GGML_AVX2=ON, GGML_AVX=ON, GGML_FMA=ON, GGML_F16C=ON, LLAMA_CURL=OFF, GGML_OPENMP=ON
    [✓] Cleaning previous build artifacts...
    [✓] Cloning llama-cpp-python v0.3.16...
    [✗] Build timed out
    [✗] Python dependencies failed
    [✓] Cleaned up compilation temp folder
    DeActivated: `.venv`
    Press any key to continue . . .

</details>

## Hard Requirements
- Windows 7-11 and/or ~Ubuntu 22-25 - Its BOTH a, Windows AND linux, program, batch for windows and bash for linux, launch dual-mode scripts.
- Python 3.9-3.13 - Requires [Python](https://www.python.org); AI warns me certain libraries wont work on Python 3.14, possibly Spacy. 
- Llm Model - You will need a Large Language Model in GGUF format, check the models section for recommendations, but for quick start I advise one like [Qwen3-4B-abliterated-GGUF](https://huggingface.co/mradermacher/Qwen3-4B-abliterated-GGUF) for testing basic operation.
- Suitable GPU - Gpu may be, Main or Compute, with VRam selection 4GB-64GB. Ideally you want the GPU to cover all model layers for fast interference.
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
- Changing height of session log height as shown in media section requires restart of GUI, to re-initialize the complicated and fragile dynamic UI through browser fake GUI interface. 
- Because of how the WT-Web interface works, there is ongoing issue with how sometimes lists of values in the configuration page are non-selectable; just select a different number in list first, then select number you want. Only way to fix this would be to drop older windows/ubuntu support.  
- The "Cancel Input/Response" button was impossible for now; Attempted most recently, 2 Opus 4.5 and 2 Grok, sessions, and added about ~=>45k, but then required, "Wait For Response" for Gradio v3 and Cancel Input for Gradio v4. Instead there is now a dummy "..Wait For Response.." button.
- Support was maintained for Windows 7-8; FastEmbed/ONNX was replaced with PyQt5 + Qt5 WebEngine. So its slower, but the plan is Windows 7-11 and Ubuntu 22-25. Other optimized projects may follow.
- Optimize context length; the chatbot will chunk data to the size of the context length, however using a max_context_length of ~128000 is EXTREMELY SLOW, and on older computers try NOT to use a context_length over ~32000. 
- The "iMatrix" models do not currently work, due to requirement of Cuda for imatrix to work. Just to save some issues for people that dont know. In other words, if the model has "i" in its label somewhere significant, then likely it is a iMatrix model, and you will need some other ChatBot that handles such things.
- VRAM dropdown, 1GB to 64GB in steps, this should be your FREE ram available on the selected card, if you are using the card at the time then this is why we have for example, 6GB for a 8GB GPU in use, safely allowing 2GB for the system, while in compute more one would use for example, the full 8GB on the 8GB GPU.
- I advise GPU can cover the Q6_K version, the Q6_K useually has negligable quality loss, while allowing good estimation of if the model will fit on a card, ie 8GB card will typically be mostly/completely cover a 7B/8B model in Q6_K compression, with a little extra to display the gradio interface, so the numbers somewhat relate with Q6_K when using same card as display.
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
There has been an overhaul, its it seems to work correctly, the next plans are...
1. Possibly more testing of newer models that are not Qwen 3.5.
2. Testing/Bugfixing on Ubuntu 22.

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

