# ![Chat-Windows-Gguf](media/project_banner.jpg)
<br>Status: Beta (Mostly there, but is also ongoing project, with options for further development)

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

- The refined Install Options menu...
```
================================================================================
    Chat-Gradio-Gguf - Gpu Options
================================================================================







    1) x64 CPU Only (No GPU Option)

    2) Vulkan GPU with x64 CPU Backend







--------------------------------------------------------------------------------
Selecton; Menu Options 1-2, Abandon Install = A: 


```

## Requirements
- Windows 7-11 and/or ~Ubuntu 22-25, Its BOTH a, Windows AND linux, program, batch for windows and bash for linux, launch dual-mode scripts.
- Llama.Cpp - Options here for, Vulkan or X64. This has been limited to what I can test (though its possible to replace llama.cpp with for eg Cuda12).
- Python => 3.9 - Requires "Python 3.9-3.13" -deepseek.
- Llm Model - You will need a Large Language Model in GGUF format, See `Models` section. Currently you are advised to use [Qwen3-30B-A3B-abliterated-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF), as this will be somewhat compitent while only requiring a 4GB GPU to store the 3B active parameters, but if you want simple testing model, then use [Dolphin3.0-Qwen2.5-0.5B-GGUF](https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-0.5B-GGUF), which will run like lightning on any GPU.
- Suitable GPU - Gpu may be, Main or Compute, with VRam 2-64GB. It must have Vulkan capability/drivers (if the installer contains files referring to Vulkan).  
- System Ram - Your system should cover the size of the model not able to be covered by the GPU (ie 2GB card with 4GB model, would require ~2.5GB additional system memory, or if no GPU then ~4.5GB spare System Ram).

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
- VRAM dropdown, 1GB to 64GB in steps, this should be your FREE ram available on the selected card, if you are using the card at the time then this is why we have for example, 6GB for a 8GB GPU in use, safely allowing 2GB for the system, while in compute more one would use for example, the full 8GB on the 8GB GPU.
- I advise GPU can cover the Q6_K version, the Q6_K useually has negligable quality loss, while allowing good estimation of if the model will fit on a card, ie 8GB card will typically be mostly/completely cover a 7B/8B model in Q6_K compression, so the numbers also relate.
- We use a calculation of `1.125`, the calculation is `TotalVRam /((ModelFileSize * 1.125) / NumLayers = LayerSize) = NumLayersOnGpu`.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.
- This project is for a chat interface, and is not intended to overlap with my other projects, `Rpg-Gradio-Gguf`, or the blueprints for, `Code-Gradio-Gguf` or `Agent-Gradio-Gguf`.
- Afterthought Countdown is, <25 characters then 1s or 26-100 charactrs then 3s or >100 lines then 5s, cooldown before proceeding, enabling cancelation relative to input.
- "Chat-Windows-Gguf" and "Chat-Linux-Gguf", is now "Chat-Gradio-Gguf", as yes, these dual-mode scripts used to be 2 different/same programs.

### New Models working recently since Revision ~0.8x (with gpt for comparrisson).
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B)      | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |

### Models working (with gpt for comparrisson). 
| Model                                  | IFEval   | BBH  /\  | MATH     | GPQA     | MuSR     | MMLU              | CO2 Cost  |
|----------------------------------------|----------|----------|----------|----------|----------|-------------------|-----------|
| Early GPT-4 (compare stats)                            | N/A      | ~50%*    | 42.2%    | N/A      | N/A      | 86.4%             | N/A       |
| Early GPT-4o (compare stats)                           | N/A      | ~60%*    | 52.9%*   | N/A      | N/A      | 87.5%*            | N/A       |
| [Qwen3-30B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF) (30B)      | N/A  | N/A   | 80.4%   | 65.8%   | 72.2%   | N/A           | N/A   |
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
| T.B.A.           | N/A      | N/A    | N/A   | N/A      | N/A      | N/A            | N/A       |

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

### Plan for the Sound/Tts system
| Mode     | TTS engine       | Audio *player* |
| -------- | ---------------- | -------------- |
| Windows  | `pyttsx3` (SAPI) | **built-in**   |
| Pulse    | `espeak-ng`      | `paplay`       |
| PipeWire | `espeak-ng`      | `pw-play`      |

# Development
1. Still additional blank line to each line it prints for "AI-Chat" responses, its not the raw output, its the actual interface.
2. Check Speech Summary. Speech needs to be 2 step process, while progress is indicated for each of the 2 phases in the status bar. it should be now intelligent, as after recieving the response, an additional prompt is sent, to, determine and select, the best contents to say, then reads that to the user. This will need to be Optimized, ie one idea, limiting context length to the batch output size for the relating iteration, then returning it to the normal number set in the json after.
3. Find new models I want to use, and that dont work yet in order to Improve model support. Currently having issues with downloading models on public wifi. Footlocker closed, next best place is ~£2.80/pint. People need to fund WiseMan-TimeLord.
4. The stop button is... `**Very Interesting**`.
5. **Safe Globals** - Standardize all global variables using safe, unique three-word labels to avoid conflicts.  
6. Web-searching is a bit iffy, I found the input "latest version of grok?" worked. Need to improve later, DDGS was hard to work with at the time due to being NEW, and most online information is for DuckDuckGo-Search library still. They are used a little differently. Investigate/upgrade.
7. Introduce a collapseable right side bar, lilke the left one but on the right, again a "C-G-G" button, that expands out to a "Chat-Gradio-Gguf" button, in the expanded panel here I want...
- a button switching right panel to visualizes the thinking/generation, in some simple method, that is somehow interesting, with a buttton under to turn "Visualize" ON/OFF, so we know its def off or on (off by default), for the next generation.
- a button switching right panel to quick settings sliders for, "Context Size" and "Batch Output", with save button, and will update relating settings in json, as well as ensure the settings are then consistent between, "the sidebar and configuration tab".
8. The "RPG Mode" will require an additional switch position on left panel for the mode "Rpg", enabling the left or right panel for Rpg settings sliders, `ai_name` and `ai_role` and `human_name` and `human_role` and `location`, in order for the user to configure the `RPG Mode`. 

## Credits
Thanks to all the following teams, for the use of their software/platforms...
- [Llama.Cpp](https://github.com/ggml-org/llama.cpp) - The binaries used for interference with models.
- [Kimi K2](https://www.kimi.com) - For most work after its release in July 2025, like Grok4 but free.
- [Grok3Beta](https://x.com/i/grok) - For much of the complete updated functions that I implemented.
- [Deepseek R1/3](https://www.deepseek.com/) - For re-attempting the things Grok3Beta was having difficulty with.
- [Perplexity](https://www.perplexity.ai) - For circumstances of extensive web research requierd.
- Python Libraries - Python libraries change, so no specific detail, but thanks to all creators of the libraries/packages currently in installer/use.

## License
This repository features **Wiseman-Timelord's Glorified License** in the file `.\Licence.txt`, in short, `if you wish to use most of the code, then you should fork` or `if you want to use a section of the code from one of the scripts, as an example, to make something work you mostly already have implemented, then go ahead`.

