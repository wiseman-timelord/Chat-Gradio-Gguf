# ![Chat-Windows-Gguf](media/project_banner.jpg)
<br>Status: Beta
- Ubuntu/Linux mode working best in latest Release version.
- Windows mode untested with latest release, currently A068 is verified all features working but A068 is windows only.
- This project is actively worked on, check back later for progress, I try to release every 2 significant updates (see, Development section and release notes).
- By "working" I mean I likely tested it in chat/websearch with, the model Dyanka 8b and the backend Vulkan and a 8GB AMD card.

## Description
Intended as a high-quality chat interface programmed towards, windows 7-11 (non-WSL) and Ubuntu 22-25, with any Gpu on GGUF models through Python ~3.9-3.13. Dynamic prompting from keywords in models enabling better, interface and prompts, for relating theme of session, With some features but no imposed, limitations or guidelines. This tool providing local, uncensored, and inference with features that enhance productivity and usability, even a comparable interface, found on premium AI services, or as far in that direction as gguf models, will allow. The configuration is intended to be intelligent, while without options reported in forums to make no difference on most models, using offline libraries where possible instead of services requiring registration, and everything is privatly run on your own hardware with your own models. 

### Features
- **Comprihensive GPU Support**: CUDA/Vulkan/ROCm, with dropdown list selection supporting multi CPU/GPU setup.
- **Research-Grade Tools**: Includes RAG, web search, chunking, THINK, and Markdown formatting, and file attachments. 
- **Common File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`, and other common formats of files.
- **Configurable Context**: Set model context to 8192-138072, and batch output to 1028-32768.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Highly Customizable UI**: Configurable; 4-16 Session History slots, 2-10 file slots, Session Log 450-1300px height, 2-8 Lines of input. 
- **Speak Summaries**: Click `Say Summary` for a special prompt for a concise spoken summary of the generated output. Text to speak uses `PyWin32`.
- **Attach Files**: Attach Files is complete raw input, there is no Vectorise Files anymore, so the files should ideally be small enough to fit in context. 
- **Collapsable Left Column**: Like one would find on modern AI interface, but with concise 3 button interface for commonly used buttons. 
- **ASynchronous Response Stream**: Separate thread with its own event loop, allowing chunks of response queued/processed without blocking Gradio UI event loop.
- **Reasoning Compatible**: Dynamic prompt system adapts handling for reasoning models optimally, ie, uncensored, nsfw, chat, code.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Correct Vulkan Installs**: If Vulkan selected, then, `Windows 7-8 = v1.1.126.0` and `Windows 8.1-11 = v1.4.3.04.1`, avoiding API issues.

### Preview
- When Requires are installed, startup looks like this in the command console (outdated)...
```
================================================================================
    Chat-Gradio-Gguf: Launcher
================================================================================

Starting Chat-Gradio-Gguf...
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

```

- The "Interaction" page, where the conversation happens...
![preview_image](media/conversation_page.jpg)

- The collapseable Left Panel on the `Interaction` page...
![preview_image](media/conversation_expand.jpg)

- The "Configuration" page - for configuration of, models and hardware, and relevant components, as well as ui customization...
![preview_image](media/configuration_page.jpg)

- The "Installation" Processes for Ubuntu Linux installation (outdated)...
```
================================================================================
    Chat-Gradio-Gguf: Installation
================================================================================

Installing Chat-Gradio-Gguf on linux...
[✓] Selected backend: GPU/CPU - Vulkan
[✓] Verified directory: data
[✓] Verified directory: scripts
[✓] Verified directory: models
[✓] Verified directory: data/history
[✓] Verified directory: data/temp
[✓] Created new virtual environment
[✓] Verified virtual environment setup
[✓] Installing Python dependencies...
Requirement already satisfied: pip in ./.venv/lib/python3.13/site-packages (25.0)
...

...
[✓] Python dependencies installation completed
[✓] Downloading llama.cpp (GPU/CPU - Vulkan)...
Downloading GPU/CPU - Vulkan: [=========================] 100% (19.7MB/19.7MB)
[✓] Extracting backend files...
Extracting: [=========================] 100% (37.0B/37.0B)
[✓] Copying Linux binaries to destination...
[✓] Copied 16 binary files
[✓] llama.cpp (GPU/CPU - Vulkan) installed successfully
[✓] Configuration file created
[✓] Chat-Gradio-Gguf installation completed successfully!

You can now run the application with Option 1 on the Bash/Batch Menu.

Deactivated: .venv
Press Enter to continue...
```

## Requirements
- Windows 7-11 - Its a Windows program, batch menu has auto-sized-layout for modern/old OS.
- Llama.Cpp - Options here for, Vulkan, ROCM, Cuda 11, Cuda 12. Llama.cpp are limiting the options now.
- Python => 3.9 - Requires "Python 3.9-3.13" -deepseek.
- Llm Model - You will need a Large Language Model in GGUF format, See `Models` section.
- Suitable GPU - Gpu may be, Main or Compute, with VRam 2-64GB. Tested with Vulkan install.  

### Instructions (W = Windows, U = Ubuntu)...
- Pre-Installation...
```
If installing with Vulkan option, you will need to have installed the `Vulkan SDK`,
If installing with CUDA option, you will need to have installed the `CUDA Toolkit`.
If installing with ROCM option, you will need to have installed the `ROCm Software Stack`.
```
- Installation...
```
1.W. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `C:\Programs\Chat-Windows-Gguf` or `C:\Program_Files\Chat-Windows-Gguf`, (try not to use spaces). 
1.U. Download a "Release" version, when its available, and unpack to a sensible directory, such as, `/media/**UserName**/Programs_250/Chat-Gradio-Gguf`, (try not to use spaces).
2.W. Right click the file `Chat-Windows-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load, then select `2` from the Batch Menu, to begin installation. You will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder. After the install completes, check for any install issues, you may need to install again if there are.
2.U. open a terminal in the created folder location, then make "Chat-Gradio-Gguf.sh" executable via `Right Mouse>Properties>Make Executable`. Then run `sudo bash ./Chat-Linux-Gguf.sh`, then select option `2. Run Installation` from the menu, this may take some time (hopefully work for you, or try it again best advice for now.).
3.W. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Chat-Windows-Gguf`. 
3.U. Having returned to the bash menu after successful install, one would use option `1. Run Main Program`, to load the gradio interface in the popup browser window.
4. You will then be greeted with the `Interaction` page, but you will first be going to the `Configuration` page. On the `Configuration` page you would configure appropriately, its all straight forwards, but remember to save settings and load model. If the model loads correctly it will say so in the `Status Bar` on the bottom od the display.
5. Go back to the `Interaction` page and begin interactions, ensuring to notice features available, and select appropriately for your, specific model and use cases.
6. When all is finished, click `Exit` on the bottom right then close browser-tabs/terminals.
```

### Notation 
- For AMD hardware, do not use models with an "iMatrix", those are for nVidia GPUs. Just to save some issues for people that dont know.
- For Vulkan installations, you must install the Vulkan SDK; I will be adding this to [Ubuntu25-TweakInstall](https://github.com/wiseman-timelord/Ubuntu25-TweakInstall), otherwise you must go here [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
- For CUDA installations, you must install the CUDA Toolkit; I added this to [Ubuntu25-TweakInstall](https://github.com/wiseman-timelord/Ubuntu25-TweakInstall), but if you have other OS, then research/instal first.
- VRAM dropdown, 1GB to 64GB in steps, this should be your FREE ram available on the selected card, it should not be the total amount the car has, unless you are running in compute mode.  
- We use a calculation of `1.125`, the calculation is `TotalVRam /((ModelFileSize * 1.125) / NumLayers = LayerSize) = NumLayersOnGpu`.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.
- This project is for a chat interface, and is not intended to overlap with my other projects, `Rpg-Gradio-Gguf`, or the blueprints for, `Code-Gradio-Gguf` or `Agent-Gradio-Gguf`.
- Afterthought Countdown is, <25 characters then 1s or 26-100 charactrs then 3s or >100 lines then 5s, cooldown before proceeding, enabling cancelation relative to input.
- "Chat-Windows-Gguf" is intended as the Windows version of [Chat-Linux-Gguf](https://github.com/wiseman-timelord/Chat-Linux-Gguf).
- This was the command to make a python script file created in windows to be compatible with linux `sed -i 's/\xE2\x80\x8B//g' scripts/interface.py`.

### Models working with v0.73 (with gpt for comparrisson).
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF           | ~76.40%   | ~36.62%   | ~48.79%   | ~8.95%    | ~15.51%   | ~37.51% (MMLU-PRO) | N/A   |
| Qwen2.5-Dyanka-7B-Preview-GGUF           | 76.40%   | 36.62%   | 48.79%   | 8.95%    | 15.51%   | 37.51% (MMLU-PRO) | 0.62 kg   |
| T.B.A.           | N/A      | N/A    | N/A   | N/A      | N/A      | N/A            | N/A       |

### Models working with version A068 (with gpt for comparrisson).
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [Lamarckvergence-14B-GGUF](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) (14B)          | 76.56%   | 50.33%   | 54.00%   | 15.10%   | 16.34%   | 47.59% (MMLU-PRO) | N/A       |
| qwen2.5-test-32b-it (32B)              | 78.89%   | 58.28%   | 59.74%   | 15.21%   | 19.13%   | 52.95%            | 29.54 kg  |
| [T3Q-qwen2.5-14b-v1.0-e3-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-GGUF) (14B)      | 73.24%   | 65.47%   | 28.63%   | 22.26%    | 38.69%   | 54.27% (MMLU-PRO) | 1.56 kg   |
| [T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/T3Q-qwen2.5-14b-v1.0-e3-Uncensored-DeLMAT-GGUF/tree/main) (14B)      | ~73.24%   | ~65.47%   | ~28.63%   | ~22.26%    | ~38.69%   | ~54.27% (MMLU-PRO) | ~1.56 kg   |
| [Qwen2.5-Dyanka-7B-Preview-GGUF](https://huggingface.co/mradermacher/Qwen2.5-Dyanka-7B-Preview-GGUF) (7B)     | 76.40%   | 36.62%   | 48.79%   | 8.95%    | 15.51%   | 37.51% (MMLU-PRO) | 0.62 kg   |
| [Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF](https://huggingface.co/mradermacher/Qwen2.5-Dyanka-7B-Preview-Uncensored-DeLMAT-GGUF) (7B)     | ~76.40%   | ~36.62%   | ~48.79%   | ~8.95%    | ~15.51%   | ~37.51% (MMLU-PRO) | ~0.62 kg   |
| [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) (7B)          | 75.83%   | 36.36%   | 48.39%   | 7.72%    | 15.17%   | 37.73% (MMLU-PRO) | N/A       |
| [Q2.5-R1-3B-GGUF](https://huggingface.co/mradermacher/Q2.5-R1-3B-GGUF) (3B)                    | 42.14%   | 27.20%   | 26.74%   | 7.94%    | 12.73%   | 31.26% (MMLU-PRO) | N/A       |

### Models (in the works)
Either, compatibility in progress or not confirmed, with gpt for comparrisson..    
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                           | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B)      | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |
| calme-3.2-instruct-78b (78B)           | 80.63%   | 62.61%   | 40.33%   | 20.36%   | 38.53%   | 70.03%            | 66.01 kg  |

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
<details>
  <summary>Keyword Info ></summary>
    
    Keywords in model label will dynamically adapt the prompt appropriately...
    - `Vanilla Chat` keywords - none of the below.
    - `Coding` keywords - "code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell".
    - `UnCensored` keywords - "uncensored", "unfiltered", "unbiased", "unlocked".
    - `reasoning` keywords - "reason", "r1", "think".
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

# Development
1. Issues with exit button.
2. Issues with attach files.
1. there is still extra lines being inputted into the output of "AI-Chat". The raw output printed to the terminal is correct, so somewhere there is a, "\n" or " ", where it doesnt belong, and is for some reasoning adding an additional blank line to each line it prints for "AI-Chat" responses.
1. there is annoying glitchy graphic objects when typing input to the model.
2. the Speech option did not work in linux.
1. windows mode requires testing/bugfixing.
2. No CPU options anymore, but we had to add back in some of the cpu code to get the gpu to work, so possibly add back in the cpu binaries as most of other code is already there.
3. Check Speech Summary, it should be now intelligent, as after recieving the response, an additional prompt is sent, to, determine and select, the best contents to say, then reads that to the user. This will need to be Optimized, ie one idea, limiting context length to the batch output size for the relating iteration. 
6. Qwen 3 and Deepseek 3, compatibility/integration/features. Make test script for both had issues, instead make individual test scripts, just getting one of them working first may be in order.
4. Need to optimize streaming output and relating displaying of text/animations. Design needs work, doesnt work how I want.
5. The stop button wont work because the button freezes during certain/all phases of generation. Or just change the message from "Cancel Sending" to "Please Wait".
6. **Safe Globals** - Standardize all global variables using safe, unique three-word labels to avoid conflicts.  
7. Web-searching is a bit iffy, I found the input "latest version of grok?" worked. Need to improve later, DDGS was hard to work with at the time due to being NEW, and most online information is for DuckDuckGo-Search library still. They are used a little differently.
7. Introduce a collapseable right side bar, lilke the left one but on the right, again a "C-G-G" button, that expands out to a "Chat-Gradio-Gguf" button, in the expanded panel here I want...
- a row with a square box, that visualizes the thinking/generation, in some simple method, that is somehow interesting, and under that, a row with a buttton to turn "Visualize" ON/OFF.
- a row with 3 boxes each with 1 stat for generation speed/whatever else is typically of interest to people.
- a row with 2 butttons, 1 to turn "Visualize" ON/OFF and 1 to turn "Statistics" on/off. Same kind of on/off buttons as web-Search/Speech/Summary.
- sliders for, context size and batch size, affecting them will affect the ones on the configuration page, and vice versa. They use the same global in temporary. we would then need to ensure that whatever settings are selected on either/both, will then be active for when the next time the user clicks send.

## Credits
Thanks to all the following teams, for their parts...
- [Llama.Cpp](https://github.com/ggml-org/llama.cpp) - The binaries used for interference with models.
- [Yake](https://github.com/LIAAD/yake) - Library used for generating the labels for the history slots.
- [Kimi K2](https://www.kimi.com) - For most work after its release in July 2025, like Grok4 but free.
- [Grok3Beta](https://x.com/i/grok) - For much of the complete updated functions that I implemented.
- [Deepseek R1/3](https://www.deepseek.com/) - For re-attempting the things Grok3Beta was having difficulty with.
- [Perplexity](https://www.perplexity.ai) - For circumstances of extensive web research requierd.


## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, `if you wish to use most of the code, then you should fork` or `if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead`.

