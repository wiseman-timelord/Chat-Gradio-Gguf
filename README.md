# ![Chat-Windows-Gguf](media/project_banner.jpg)

### Status
Beta - Working fine on v0.97.1...

### Current News
- Proper testing/fixing is going on in order to, complete the project and make the video, its working 100% correctly now, processing now even has dynamic progress bar.

## Description
Intended as a high-quality chat interface with wide hardware/os support, windows 7-11 (WSL not required) and Ubuntu 22-25, with any Gpu on GGUF models through Python ~3.9-latest. Dynamic prompting from keywords in models, as well as dynamic buttons on the interface, all on local models so no imposed, limitations or guidelines (model dependent). This tool providing a comparable interface found on premium AI services, with, web-search and tts. The configuration tab is intended to be intelligent, while without options reported in forums to make no difference on most models. The program using offline libraries where possible instead of online services or repeat download or registration. 

### Features
- **Edge-WebView Custom Browser**: The interface uses Edge-WebView with Gradio, this, looks better and means your cookies are def safe.  
- **Comprihensive GPU Support**: Vulkan, with dropdown list in configuration selection supporting multi CPU/GPU setup.
- **Research-Grade Tools**: Includes RAG, web search, chunking, THINK, and Markdown formatting, and file attachments. 
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
- The "Interaction" page (Edge-WebView) where conversation happens, with the "Command Prompt" output in the background...
![preview_image](media/conversation_page.jpg)

- The collapseable Left/Right Panel on the `Interaction` page (click the C-G-G button)...
![preview_image](media/conversation_expand.jpg)

- The dynamic progress indication replaces the user input box upon submission...
![preview_image](media/dynamic_progress.jpg)

- The "Configuration" page - for configuration of, models and hardware, and relevant components, as well as ui customization...
![preview_image](media/configuration_page.jpg)

- Startup looks like this in the terminal/console (outdated)...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf: Launcher
    ===============================================================================
    
    Starting Chat-Gradio-Gguf..
    Activated: .venv
    `main` Function Started.
    Config loaded
    Finding Models: ...2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF
    Models Found: ['Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT.Q6_K.gguf']
    Script mode `linux` with backend `Vulkan`
    Working directory: ...s_250/Chat-Gradio-Gguf/Chat-Gradio-Gguf-A069
    Data Directory: .../Chat-Gradio-Gguf/Chat-Gradio-Gguf-A069/data
    Session History: ...adio-Gguf/Chat-Gradio-Gguf-A069/data/history
    Temp Directory: ...-Gradio-Gguf/Chat-Gradio-Gguf-A069/data/temp
    CPU Configuration: 12 physical cores, 24 logical cores
    
    Configuration:
      Backend: Vulkan
      Model: Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT.Q6_K.gguf
      Context Size: 49152
      VRAM Allocation: 8192 MB
      CPU Threads: 20
      GPU Layers: 0
    
    Launching Gradio Interface...
    * Running on local URL:  http://127.0.0.1:7860
    * To create a public link, set `share=True` in `launch()`.

</details>

- The updoming Install Options menu, regular download and/or compile if having pre-installed tools...
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf - Gpu Options
    ===============================================================================
    
    
    
        1) Download CPU Binaries / Download CPU Wheel
    
        2) Compile CPU Binaries / Compile CPU Wheel
    
        3) Download Vulkan Bin / Download CPU Wheel
    
        4) Download Vulkan Bin / Download CPU Wheel (Forced)
    
        5) Download Vulkan Bin / Compile Vulkan Wheel
    
        6) Compile Vulkan Binaries / Compile Vulkan Wheel
    
    
    
    
    -------------------------------------------------------------------------------
    Selecton; Menu Options 1-6, Abandon Install = A:

</details>

- Installation processes (after building Vulkan binaries/Wheel in limited terminal/console buffer)..
<details>
    
          ...shortened...
      Installed llama-save-load-state.exe
      Installed llama-server.exe
      Installed llama-simple-chat.exe
      Installed llama-simple.exe
      Installed llama-speculative-simple.exe
      Installed llama-speculative.exe
      Installed llama-tokenize.exe
      Installed llama-tts.exe
      Installed llama-vdot.exe
    [V] Binary compilation complete
    Optimizations enabled: AVX2, FMA, F16C (50% less RAM)
    [V] Configuration file created
    
    Generated configuration:
      Backend: VULKAN_VULKAN
      Vulkan Available: True
      VRAM: 8192 MB
      Context: 32768
      llama-cli: .\data\llama-vulkan-bin\llama-cli.exe
    [V] Installation complete!
    
    Run the launcher to start Chat-Gradio-Gguf
    
    DeActivated: `.venv`
    Press any key to continue . . .

</details>

- Validation of Install (script needs checking/updating since compiling update)..
<details>
    
    ===============================================================================
        Chat-Gradio-Gguf: Library Validation
    ===============================================================================
    
    Running Library Validation...
    Activated: `.venv`
    === Chat-Gradio-Gguf Validator (WINDOWS) ===
    
    === Directory Validation ===
      V data
      V scripts
      V models
      V history
      V temp
      V vectors
      V fastembed_cache
    
    === Configuration Validation ===
      V Config valid (Backend: Vulkan)
      V llama-cli configured: llama-cli.exe
    
    === Backend Binary Validation ===
      V llama-cli found: llama-cli.exe
    
    === Core Library Validation ===
      V gradio
      V requests
      V pyperclip
      V spacy
      V psutil
      V ddgs
      V newspaper3k
      V langchain-community
      V faiss-cpu
      V langchain
      V pygments
      V lxml
      V pyttsx3
      V onnxruntime
      V fastembed
      V tokenizers
      V xllamacpp
      V pywin32
      V tk
    
    === Optional Library Validation ===
      V PyPDF2 (optional)
      ? python-docx (not installed - optional)
      V openpyxl (optional)
      ? python-pptx (not installed - optional)
    
    Note: 2 optional packages missing (text-only fallback will be used)
    
    === spaCy Model Validation ===
      V en_core_web_sm model available
    
    === FastEmbed Model Validation ===
      V FastEmbed model verified
    
    === Validation Summary ===
      V All validations passed successfully!
    
    Your installation is ready to use.
    DeActivated: `.venv`
    Press any key to continue . . .
    
</details>

## Requirements
- Windows 7-11 and/or ~Ubuntu 22-25, Its BOTH a, Windows AND linux, program, batch for windows and bash for linux, launch dual-mode scripts.
- Llama.Cpp - Options here for, Vulkan or X64. This has been limited to what I can test (though its possible to replace llama.cpp with for eg Cuda12).
- Python => 3.9 - Requires "Python 3.9-3.13" -deepseek.
- Llm Model - You will need a Large Language Model in GGUF format, See `Models` section. Currently you are advised to use [Qwen1.5-0.5B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF), as this will be super-fast and confirm basic operation, otherwise see models section.
- Suitable GPU - Gpu may be, Main or Compute, with VRam selection 756MB-64GB. It must have Vulkan capability/drivers (if the installer contains files referring to Vulkan).  
- System Ram - Your system should cover the size of the model not able to be covered by the GPU (ie, 4GB card for 4B model in Q6_K, 16GB card for 16b model in Q6_K).

## Additional Requirements 
For compile options; If on PATH, ask AI how to check its on path, and as applicable fix...
- MSVC 2017-2019 - MSVC with option Desktop Development enabled during install.
- MS Build Tools - Also for building, ensure its on PATH.
- Git - Github Program for cloning the sources from github, ensure its on PATH. 

### Instructions (W = Windows, U = Ubuntu)...
- Pre-Installation...
```
If installing with Vulkan option, you will need to have installed the `Vulkan SDK`,
```
- Installation...
```
1.W. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `C:\Programs\Chat-Windows-Gguf` or `C:\Program_Files\Chat-Windows-Gguf`, (try not to use spaces). 
1.U. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `/media/**UserName**/Programs_250/Chat-Gradio-Gguf`, (try not to use spaces).
2.W. Right click the file `Chat-Windows-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load, then select `2` from the Batch Menu, to begin installation. You will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder. After the install completes, check for any install issues, you may need to install again if there are.
2.U. open a terminal in the created folder location, then make "Chat-Gradio-Gguf.sh" executable via `Right Mouse>Properties>Make Executable`. Then run `sudo bash ./Chat-Linux-Gguf.sh`, then select option `2. Run Installation` from the menu, this may take some time (hopefully work for you, or try it again best advice for now.).
- There are now 6 install options for download/compile...
1) Download CPU Binaries + Download  CPU Wheel (CPU_CPU)
2) Compile CPU Binaries + Compile CPU Wheel (CPU_CPU)
3) Download Vulkan Binaries + Download Cpu Wheel (VULKAN_CPU)
4) Download Vulkan Binaries + Download Cpu Wheel  (Forced) (VULKAN_CPU) 
5) Download Vulkan Binaries + Compile Vulkan Wheel (VULKAN_VULKAN)
6) Compile Vulkan Binaries + Compile Vulkan Wheel   (VULKAN_VULKAN)
3.W. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Chat-Windows-Gguf`. 
3.U. Having returned to the bash menu after successful install, one would use option `1. Run Main Program`, to load the gradio interface in the popup browser window.
4. You will then be greeted with the `Interaction` page, but you will first be going to the `Configuration` page. On the `Configuration` page you would configure appropriately, its all straight forwards, but remember to save settings and load model. If the model loads correctly it will say so in the `Status Bar` on the bottom od the display.
5. Go back to the `Interaction` page and begin interactions, ensuring to notice features available, and select appropriately for your, specific model and use cases.
6. When all is finished, click `Exit` on the bottom right then close browser-tabs/terminals.
```

### Notation 
- Optimize context length; the chatbot will chunk data to the size of the context length, however using a max_context_length of ~128000 is EXTREMELY SLOW, and on older computers try NOT to use a context_length over ~32000. 
- The "iMatrix" models do not currently work, due to requirement of Cuda for imatrix to work. Just to save some issues for people that dont know.
- For Vulkan installations, you must install the Vulkan SDK, it may come with your graphics card, otherwise you must go here [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
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
| [Huihui-gpt-oss-20b-BF16-abliterated-v2-GGUF](https://huggingface.co/mradermacher/Huihui-gpt-oss-20b-BF16-abliterated-v2-GGUF) (20B)      | ~84.1%  | ~58.1%   | ~96.0-98.7%   | ~71.5%    | ~42.5%    | ~85.3%           | N/A   |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B-A3B)      | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |
| [Lamarckvergence-14B-GGUF](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) (14B)          | 76.56%   | 50.33%   | 54.00%   | 15.10%   | 16.34%   | 47.59% (MMLU-PRO) | N/A       |
| qwen2.5-test-32b-it (32B)              | 78.89%   | 58.28%   | 59.74%   | 15.21%   | 19.13%   | 52.95%            | 29.54 kg  |
| [T3Q-qwen2.5-14b-v1.0-e3-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-GGUF) (14B)      | 73.24%   | 65.47%   | 28.63%   | 22.26%    | 38.69%   | 54.27% (MMLU-PRO) | 1.56 kg   |
| [T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF/tree/main) (14B)      | ~73.24%   | ~65.47%   | ~28.63%   | ~22.26%    | ~38.69%   | ~54.27% (MMLU-PRO) | ~1.56 kg   |
| [Qwen2.5-Dyanka-7B-Preview-GGUF](https://huggingface.co/mradermacher/Qwen2.5-Dyanka-7B-Preview-GGUF) (7B)     | 76.40%   | 36.62%   | 48.79%   | 8.95%    | 15.51%   | 37.51% (MMLU-PRO) | 0.62 kg   |
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

### New Models of interest, that may/will, be downloaded by "W-T" and be compatible with future versions of "C-G-G"... 
- [Apriel-1.5-15b-Thinker-GGUF](https://huggingface.co/jobist/Apriel-1.5-15b-Thinker-Q5_K_M-GGUF) (15B) (Image Reading)
- [InternVL3_5-GPT-OSS-20B-A4B-Preview-gguf](https://huggingface.co/QuantStack/InternVL3_5-GPT-OSS-20B-A4B-Preview-gguf) (20b-A4B)
- [Qwen3-VL-8B-Thinking-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-8B-Thinking-GGUF) (8B) (Image Reading) (256,000 context window)
- [Qwen3-VL-4B-Thinking-GGUF](https://huggingface.co/NexaAI/Qwen3-VL-4B-Thinking-GGUF) (4B) (Image Reading) (256,000 context window)
- 
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
│ └── interface.py
│ └── models.py
│ └── prompts.py
│ └── settings.py
│ └── temporary.py
│ └── utlity.py
```
- Installed/Temporary files...
```
project_root/
├── data/
│ └── persistent.json
├── data/vectors/
└─────── *
├── data/temp/
└────── *
├── data/history
└────── *
├── .venv/
└────── *
```

### Plan for the Sound/Tts system
| Mode     | TTS engine       | Audio *player* |
| -------- | ---------------- | -------------- |
| Windows  | `pyttsx3` (SAPI) | **built-in**   |
| Pulse    | `espeak-ng`      | `paplay`       |
| PipeWire | `espeak-ng`      | `pw-play`      |

# Development
- Remember: This project is for a chat interface, and is not intended to overlap with other blueprints/projects, `Rpg-Gradio-Gguf` or `Code-Gradio-Gguf` or `Agent-Gradio-Gguf`. This Program is also intended to be basic, in order for people to be able to use it as a framework for more complicated AI systems, or even my other projects.
1. (issues with truncation of input) If context size is loaded with model at 8k, then modified to 64k, then I try to input ~50k of data, it then tries to input 50k into 8k, and produces an error. Either, there is something that is not updating or when the context size is altered, at the point of "Save Settings" we need to reload the model.
1. Still additional blank line to each line it prints for "AI-Chat" responses under windows but not linux, and not certain browsers, its not the raw output, its the actual interface, it seemed fine on laptop though, so possibly this is a browser issue or my desktop pc setup, or a driver issue.
3. **Safe Globals** - Standardize all global variables using safe, unique three-word labels to avoid conflicts.  
4. Web-searching is a bit iffy, I found the input "latest version of grok?" worked. Need to improve later, DDGS was hard to work with at the time due to being NEW, and most online information is for DuckDuckGo-Search library still. They are used a little differently. Investigate/upgrade.

## Credits
Thanks to all the following teams, for the use of their software/platforms...
- [Llama.Cpp](https://github.com/ggml-org/llama.cpp) - The binaries used for interference with models.
- [Claude.AI](https://claude.ai/chat) - It is the best, some people find the necessity for "Shift+Enter" an issue.
- [Kimi K2](https://www.kimi.com) - For most work after its release in July 2025, like Grok4 but free.
- [Grok3Beta](https://x.com/i/grok) - For much of the complete updated functions that I implemented.
- [Deepseek R1/3](https://www.deepseek.com/) - For re-attempting the things Grok3Beta was having difficulty with.
- [Perplexity](https://www.perplexity.ai) - For circumstances of extensive web research requierd.
- Python Libraries - Python libraries change, so no specific detail, but thanks to all creators of the libraries/packages currently in installer/use.

## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, `if you wish to use most of the code, then you should fork` or `if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead`.

