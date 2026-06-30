# ![Chat-Gradio-Gguf](media/project_banner.jpg)
Status - Troubled.
- Currently v2.0xx.xx - Seems to work great on Qwen v3.0-v3.6 and some other models, see list, but stick to Mem-Lock loading in the config, unless you intend to use one-shot, which will require restarting if you want to start a new session. One-Shot is being fixed.
- Previously v1.xx - Linux installs/use was updated/confirmed working for linux in some versions of v1.
- Previously v0.xx - there are some interesting quirks to some of them, some of them have rpg elements before I created Rpg-Gradio-Gguf.  

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
- **Text To Speech**: Corqui-TTS for realistic reading of output, where output is filtered for all symbols/tags/thinking appropriately.
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
- The Inference page, mid-response, showing both side panels expanded, and dynamic progress indication replacing the user input box upon submission (Windows v2.01.00)...
![image_missing](media/dynamic_progress.jpg)

- The Config page, for configuration of, Hardware, TTS, Models (v1.02.1)...
![image_missing](media/config_page.jpg)

- The Settings page, for configuration of, GUI, Filters (v1.02.1)...
![image_missing](media/settings_page.jpg)

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

- The combined Info/Install menu, choose 3/4 if System Detections are as shown (v2.0.0)...
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

## Hard Requirements
- Windows 10-11 and/or Ubuntu 24-25 - Its BOTH programs in merged scripts, launching via, batch for windows and bash for Ubuntu. Note: Ubuntu has not been tested since v2, it likely will not work currently (check v1 releases for mentions of Ubuntu/Linux testing/work being done, but it will be older models only).
- Python ~3.11-~3.13 - Requires [Python](https://www.python.org); AI warns me certain libraries wont work on Python 3.14, possibly Spacy. 
- Llm Model - You will need a Large Language Model in GGUF format, check the models section for recommendations, but for quick start I advise one like a Qwen3 4B GGUF model such as [Qwen3-4B-abliterated-GGUF](https://huggingface.co/mradermacher/Qwen3-4B-abliterated-GGUF) for testing basic operation or fast responses, and a Qwen 30b A3B GGUF model for proper testing.
- Suitable GPU - Gpu may be, Main or Compute, with VRam selection 4GB-96GB. Ideally you want the GPU to cover all model layers for fast interference.
- System Ram - Your system ram must cover, the curren load and the size of the model layers not able to be covered by the GPU, plus smaller models like the embeddings model, plus the wheel if not built for Vulkan.

### Building Requirements 
For compile options; If on PATH, ask AI how to check its on path, and as applicable fix...
- [MSVC++ 2019-2022](https://visualstudio.microsoft.com/vs/older-downloads/) - MSVC with option Desktop Development enabled during install.
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
- Test Prompts...
```
LOCAL: How much wood could a wood chuck chuck if a wood chuck could chuck wood?!
WEB SEARCH: Produce web research upon recent events relating to the recent events in the war in the middle-east, then compile a timeline report with most significant events for the most recent 28 days from the current date.
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
- The idea of v2 limiting compatibility being people with older OS likely not target audience for inference, and through streamling and simplifying complexity, to make possible additional complexity to upgrade. 
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
- If you installed with compile option, then you will be able to use the newer models, otherwise with non-compile install will work with models from ~6 months ago, but dont quote me on that. This is due to a versioning difference between, the [latest pre-built install](https://github.com/eswarthammana/llama-cpp-wheels/releases) and compiling the latest llama.cpp wheel from source. If not compiling, then the model handling will be limited to GLM4.7/Qwen3 level models.  
- Chat-Gradio-Ggud is currently being programmed towards these models, and performance for the intended models is reported to be like this (no good ~30b models for GLM 5.1 yet, currently GLM 4.7 seems consistently bad compared to Qwen, regardless of better scores people say were internal/faked)...

| Model                                      | IFEval                  | BBH            | MATH / MATH-500 / AIME          | GPQA / Diamond       | MuSR       | MMLU / MMLU-Pro              | Notes / Other |
|--------------------------------------------|-------------------------|----------------|---------------------------------|----------------------|------------|------------------------------|---------------|
| Early GPT-4 (ref)                          | N/A                     | ~50%*          | 42.2%                           | N/A                  | N/A        | 86.4%                        | Reference |
| Early GPT-4o (ref)                         | N/A                     | ~60%*          | 52.9%*                          | N/A                  | N/A        | 87.5%*                       | Reference |
| **GLM-4.7-Flash (30B-A3B MoE)**            | High (~83-91% est.)     | Strong         | AIME 2025: 91.6%                | 75.2%                | N/A        | ~80-84% / Strong Pro         | Excellent coding/agentic; SWE-bench Verified 59.2%; strong tool-calling |
| **Qwen3-30B-A3B (MoE)**                    | ~91.6%                  | ~62-72%        | 80.4%+ / AIME ~70.9%            | 65.8%+               | 72.2%      | High 70s-82%                 | Strong efficient MoE |
| **Qwen3-32B**                              | 93.7% (non-thinking) / 91.0% (thinking) | High (~84-86%) | Strong (MATH-500 96.1% thinking)| 66.8% (thinking)     | N/A        | ~83-87% / 72.7-79.8% Pro     | Excellent dense ~32B performer |
| **Qwen3.5-27B**                            | 93.9% / 95.0%           | Strong         | Very strong (high AIME/MATH)    | 82.8-85.5% Diamond   | N/A        | 82-86% / 83.7-86.1% Pro      | Outstanding mid-size balance |
| **Qwen3.5 ~32-35B class (e.g. 35B-A3B)**  | 93-94%+                 | Strong         | Excellent (AIME high 80s-90s)   | ~82-87% Diamond      | High       | 84-87% / 84-87% Pro          | Strong MoE variants in this size |
| Gemma 4 31B (Dense)                        | High                    | Strong         | Excellent (~89% AIME)           | ~84.3% Diamond       | N/A        | 87.1% / 85.2% Pro            | Strong math/coding |
| Gemma 4 26B MoE                            | High                    | Strong         | Strong (~88% AIME)              | ~82% Diamond         | N/A        | ~82.7% / 82.6% Pro           | Efficient variant |
| gpt-oss-20b                                | 84.1%                   | 58.1%          | 96.0-98.7%                      | 71.5%                | ~42.5%     | 85.3%                        | Exceptional MATH |
| Minstral 3 / Mistral Large 3 family        | High 80s-90s            | Competitive    | Strong (~85% AIME)              | ~80%+                | N/A        | ~81%+ / Pro competitive      | Good all-rounder |
| Granite 4.1-30B (Instruct)                 | 89.65%                  | 83.74%         | Strong (GSM8K etc.)             | 45.76%               | N/A        | 80.16% / 64.09% Pro          | Excellent tool-calling |
| Granite 4.1-8B                             | ~87%                    | ~80.5%         | Good                            | ~42%                 | N/A        | ~73.8% / 56% Pro             | Compact & efficient |

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

# Development for v2
v2 is now in Release stage, but there is still work planned...
- LONG-RUNNING ISSUE: The output text has double blank lines inserted. Output text needs to be fixed, or we need to find alternate interface/browser solution. The long running issue of additional lines between each line is a long running problem, but we are now getting double blank lines, which may mean the filter is not working correctly. Surely one of the AI can by now say, "this is the issue" to the additional blank lines thing. Gur. As a result of recent updates both user input box and AI chat output seems to be having blank lines inbetween each line of text. This needs investigating urgently.
- The project now aims to support these specific models shown below, but do so well. Qwen3 and GLM 4.7 level models should be kept, in order to keep compatibility with non-compile install options. The main program needs to support the models shown below, thereabouts, we need complete handling for each one. This includes, if they are thinking, then what is the end thinking tag? `</THINK>`, `Answer :`, "Final Response:", before final response, in order for the thinking phase to end correctly, and other such handling quirks that we have for models in the list, but that also only those models, including variants of those models, eg abliterated, huihui, etc....
```
GLM
Qwen
Gemma
gpt oss
minstral
granite
Llama
Kimi
Deepseek
```
- STT - We could have a STT button in the tools section, enabling the input box to switch to a sample display and a button, the user would click and hold the button to record, and then let go of mouse when they finished recording, and then the wave appear in the box, then AI translate this into words, these words are then shown/editable in the text input box, and the wave record box will hide, but there will be a new button at bottom of text input when STT is enabled, to switch back to the STT Recording box and hide the text box again, so the user can re-record (blanking the previous recorded text upon pressing record). if the user selects STT again to disable it then, the wave box will hide, the text input box will be shown, and the Re-Record button will be hidden. Hmm. Is this the best way to do this? needs a brainstorm. 

### Development (reasoning notes)
- Think/NoThink button turned out to be bad idea, because quantized highly trained models are designed to either be a Thinking or Non-Thinking model, and thinking models simply, do not work well without or are unable to stop using, thinking mode. While one could say it would be useful to have a thinking mode button for non-thinking models, I consider current non-thinking models to be Nieche or low performers.
- Image reading (this would additionally require vllm, which could switch for such iterations involving image reading). So, the spanner in the works is we would need VLLM, but even then AI has ALWAYS failed at implementing this so far. If attempted again, then start with test scripts.

### Development for v1 
- Best way to do a Legacy compatible version will be cloned from main, and then add compatibility back in and release that as the final v1 version. 

## Credits
Thanks to all the following teams for their assistance/comtributions...
- The Paid/Free AI platforms used: [Claude Sonnet/Opus](https://claude.ai/chat), [Kimi K2/K2.5](https://www.kimi.com), [XAI Grok](https://x.com/i/grok), [Deepseek R1/3](https://www.deepseek.com/), [Perplexity](https://www.perplexity.ai) - 
- The Python libraries included: [Gradio](https://gradio.app/), [PyTorch](https://pytorch.org/), [llama-cpp-python](https://llama-cpp-python.readthedocs.io/), [sentence-transformers](https://sbert.net/), [LangChain](https://www.langchain.com/), [spaCy](https://spacy.io/), [PyQt6](https://www.riverbankcomputing.com/software/pyqt/), [Coqui TTS](https://github.com/coqui-ai/tts), [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/), [Numpy](https://numpy.org/), [Transformers](https://huggingface.co/docs/transformers/), [faiss-cpu](https://github.com/facebookresearch/faiss), [Aiohttp](https://aiohttp.readthedocs.io/), [Newspaper4k](https://github.com/codelucas/newspaper).

## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, if you wish to use most of the code, then you should fork, or, if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead, but, do not claim, my vanilla or =>50% of my, releases as your own work`.

