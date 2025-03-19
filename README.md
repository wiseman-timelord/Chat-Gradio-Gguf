# Chat-Gradio-Gguf
![banner_image](media/project_banner.jpg)
<br>Status: Alpha - See `Development` section..

## Description
Intended as a high-quality chat interface programmed towards windows 10 non-WSL, with any Cpu/Gpu on GGUF models. Dynamic modes enabling correct, interface and prompts, for relating theme of sessions, With the latest advancements, and no imposed, limitations or guidelines. This tool providing local, uncensored, and inference with features that enhance productivity and usability, even a comparable interface, found on premium AI services, or as far in that direction as, Gradio and Grok3Beta, will allow. The configuration is without options reported to make no difference on most models, ensuring a comprehensive yet streamlined experience. Capable of all things possible through simple scripts and awesome libraries and modern GGUF LLMs.

### Features
- **Comprihensive CPU/GPU Support**: CPUs x64-CPU/AVX2/AVX512 and GPUs AMD/nVIDIA/Intel, with dropdown list selection supporting multi CPU/GPU setup.
- **Research-Grade Tools**: Includes RAG, web search, chunking, summarization, TOT, no-THINK, and code formatting, and with file attachments. 
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Common File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`, and other common formats of files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **FAISS Vector Database**: Stores numerical vectors, and retrieves based on proximity in meaning, enabling pulling context from documents.
- **Highly Customizable UI**: Configurable; 10-20 Session History slots, 2-10 file slots, Session Log 400-550px height, 2-8 Lines of input. 
- **Afterthought Countdown**: If, =>10 lines then 5s or 5-10 lines then 3s or <5 lines then 1s, wait for you to cancel send and return to edit, saving on mis-prompts and waits.
- **Attach or Vectorise**: Optimally for the mode, Attach Files is complete raw input, while Vectorise Files is relevance selected snippets from potentially larger source. 
- **Collapsable Left Column**: Like one would find on modern AI interface. 
- **ASynchronous Response Stream**: Separate thread with its own event loop, allowing chunks of response queued/processed without blocking Gradio UI event loop.

### Preview
- The "Interaction" page with Panel Mode selector, and other features or enhancements...
![preview_image](media/conversation_page.jpg)

- The "Configuration" page - for configuration of, models and hardware, and relevant components, as well as ui customization...
![preview_image](media/configuration_page.jpg)

- Wide Llama.Cpp support in the installer, thanks to latest ~128k ai systems, no longer having to streamline such things...
```
========================================================================================================================
    Chat-Gradio-Gguf: Install Options
========================================================================================================================



 Select the Llama.Cpp type:

    1. AVX2 - CPU Only - Must be compatible with AVX2

    2. AVX512 - CPU Only - Must be compatible with AVX512

    3. NoAVX - CPU Only - For older CPUs without AVX support

    4. OpenBLAS - CPU Only - Optimized for linear algebra operations

    5. Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU

    6. Kompute - GPU/CPU - Experimental Vulkan with with CPU fallback

    7. CUDA 11.7 - GPU/CPU - For CUDA 11.7 GPUs with CPU fallback

    8. CUDA 12.4 - GPU/CPU - For CUDA 12.4 GPUs with CPU fallback



========================================================================================================================
 Selection; Menu Options = 1-8, Exit Installer = X:

```

## Requirements
- Windows 10/11 - Its a Windows program, it may be linux compatible later (not now).
- Llama.Cpp - Options here for, Avx2, Vulkan, Kompute, Cuda 11, Cuda 12.
- Python => 3.8 (tested on 3.11.0) - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.
- Llm Model - You will need a Large Language Model in GGUF format, See `Models` section.
- Suitable CPU/GPU - Gpu may be, Main or Compute, with VRam 2-64GB, testing on rx470 in Compute.  

### Instructions
1. Download a "Release" version, when its available, and unpack to a sensible directory, like, `C:\Program_Filez\Chat-Gradio-Gguf` or `C:\Programs\Chat-Gradio-Gguf`. 
2. Right click the file `Chat-Gradio-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load, then select `2` from the Batch Menu, to begin installation. You will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder. After the install completes, check for any install issues, you may need to install again if there are.
3. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Chat-Gradio-Gguf`. You will then be greeted with the `Interaction` page, but you will first be going to the `Configuration` page. 
4. On the `Configuration` page you would configure appropriately, its all straight forwards, but remember to save settings and load model. If the model loads correctly it will say so in the `Status Bar` on the bottom od the display.
5. Go back to the `Interaction` page and begin interactions, ensuring to notice features available, and select appropriately for your, specific model and use cases.
6. When all is finished, click `Exit` on the bottom right and/or close browser-tabs/terminals, however you want to do it. 

### Notation
- VRAM dropdown, 1GB to 32GB in steps, this should be your FREE ram available on the selected card.
- We use an incorrect calculation of I think `1.1` currently, but the safe calculation was `TotalVRam /((ModelFileSize * 1.1875) / NumLayers = LayerSize) = NumLayersOnGpu`.
- Most GGUF text models will work, keep in mind the applicable keywords shown in `Model Label Keywords` section, for enhancement detection.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.

### Model label/name Keywords...
Keywords in model label will dynamically adapt the prompt appropriately...
- `Vanilla Chat` keywords - none of the below.
- `Coding` keywords - "code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell".
- `UnCensored` keywords - "uncensored", "unfiltered", "unbiased", "unlocked".
- `reasoning` keywords - "reason", "r1", "think".

# Models
You will of course need to have a `*.Gguf` model, use an iMatrix version of the same models if you have nVidia hardware. Here are some of the models used to test the program.. 
1. For non-iMatrix 14B models...
- [Lamarckvergence-14B-GGUF](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) - Best ~14B model, 57th overall beating most ~70B.
2. For non-iMatrix ~8B models.
- Llama [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) 8GB cutdown but somewhat compitent - Uncensored.
- [Ninja-v1-NSFW-RP-GGUF](https://huggingface.co/mradermacher/Ninja-v1-NSFW-RP-GGUF) - Most downloaded RP NSFW on huggingface at the time.
4. Models that do not work...
- Qwen based models, such as, [DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF) and [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) and [DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF)
- Instruct based models, such as [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) and [Llama-3.2-3B-Instruct-uncensored-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF).

## File Structure
- Core Project files...
```
project_root/
│ Chat-Gradio-Gguf.bat
│ requisites.py
│ launcher.py
├── media/
│ └── project_banner.jpg
├── scripts/
│ └── interface.py
│ └── models.py
│ └── prompts.py
│ └── temporary.py
│ └── utlity.py
```
- Installed/Temporary files...
```
project_root/
├── data/
│ └── persistence.json
├── data/vectors/
└─────── *
├── data/temp/
└────── *
├── data/history
└────── *
├── .venv/
└────── *
```

## Gen 1 Development
With regards to the current version of the program...
1. It is now Chat only, and streamlined as a result. This is all now do-able and has mostly/all already been working in earlier versions, just need to fix the features based on what I already figured out. To note, 6. Rpg rpg overlapping with `Rpg-Gradio-Gguf`, so removed Rpg elements. To note, Coder mode was not possible without dual model, due to needing a text to instruct conversion, so removed until =>Gen2. This will require re-brand to `Chat-Gradio-Gguf`, inline with, `Rpg-Gradio-Gguf` and new project blueprint `Code-Gradio-Gguf`, 3 programs, not 1 that struggles to do all.
1. Continue work on prompting system until it works soundly, check all model keywords themes with enhancement combination (note text to speak uses `PyWin32`). 
5. Output is now brokwn. It seemed to work before as a subprocess while printing output, and stream when its the final response or for direct answers. We want streaming raw text input/output to the command window, while keeping it as subprocess, for the gradio display to update correctly.

## Gen => 2 Development.
Will have to figure out if I am making a Coder upgrade, or just making `Code-Gradio-Gguf` lager, hence, `Chat-Gradio-Gguf` would become `Chat-Gradio-Gguf` again, and just keep the programs separate. Obviously I already have a semi-working version of `Rpg-Gradio-Gguf`. Putting all 3 in the same program, is causing a headache when it comes to updates. But, here are the old note With regards to the features expected of future generations of the program...
1. Agentic operation...with multi-phase interactions, for such multiple prompt interactions, I want such workflows to be visualized, such as `DECIDING ON ATTACHMENTS -> (processing user input) -> (generating final response) `, then when it is processing the user input then the line would be replaced with `(deciding on attachments) - PROCESSING USER INPUT -> (generating final response) `, then when it is generating the final response then the line would be replaced with `(deciding on attachments) - (processing user input) -> GENERATING FINAL RESPONSE`, and also variants of this multi-phase visualization for promcesses involved, should be done, for visualised feed back, enhancing user experience. We already have the `█` in the status bar for progress indication in the status bar, that is great. Either way, such multi-part processes involving prompting will be represented as such, and simlarly the `█` should be used optimally in the status bar for progress in each phase, therein, so the only option is to then re-think each operation mode, and make them more effectively use multiple interactions for each response, but also visualization would include library usage for things such as, Yake or Zipfile, process, around 1-2 usage of model, as I think is how mostly the implementations currently are. So it is already somewhat agentic, and this should be figured out, and visualized as such. This visualization of phases or stages within interactions will require continual updade, but also act as an explamation or map of what is going on within the options of processes available. Plan it all out, determine, what is already present and main events within the chain of interactions between, libraries used and model interactions, to make up different stages for the visualization, that should always have i am guessing 2-6 main phases depending upon their complexity, we do not want simple tasks not involving libraries or models to be a phase.
1. re-attempt left column collapse switch with two images in `.\media` instead of text, so as to be in-line with quality graphic I made for this readme.md. 
1. Research local and online AI programs, what kinds of enhancements are common, what would be simple to implement. Someone mentioned Manu, does some kind of webscrape or internet research, producing reports.
1. Coder mode must be agentic =m requiring 2 models. 1 =>8b model for management of tasks and 1 =>8b model instruct for generation of code. Files provided in coder mode,  would NOT be in the vector database like it is in chat mode, but instead be directly and selectively sent to the code model, that thinking model would have to assess which scripts are likely relevant and produce a list and the correct instruction for each task and stage. There would also have to be the editable scrollable prefabs for each one...
```
INFORMATON:
We are working on my project ProjectName, it is a progtam written in InsertProgrammingLanguage v#.## intended to be run on InsertOperatingSystem.

STRUCTURE:
(Information about directory structure, ie core directory list)

INSTRUCTION:
(State issue with the code or what needs to be created/redacted).
(State process of investigaction and relevant information).

SECTION(s):
(Insert code snippets here)

RESOURCE(s):
(Detail things provided, remember to mention Files attached if so.)

CONSIDERATION(s):
Print complete updated function(s) for specifically the functions we are updating.
or
Print the complete updated script.
```
...something like that, this also prompts ability to change the size of the user input/session log, via some kind of draggable border that makes one larger while the other smaller, and vice versa, or if that does not exist in gradio, then maybe a slider that could be at the top of the screen, and controll linearly the proportions of each of the settings ranges for the sizes of both being the maximums and 10px or 1 line being the minimum. However its done best. So `Session Log Height` would be labeled `Max Log Height`, and `Input Lines` would become `Max Input Lines`.
1. **Safe Globals** - Standardize all global variables using safe, unique three-word labels to avoid conflicts.  
2. **Cross-Platform Scripting:** Introduce a unified script (`Chat-Gradio-Gguf.sh`) to support both Linux and Windows environments, and adapt scripts appropriately.  
3. **User Interface and Notification Enhancements:**  
   - Implement verbose, clear, and concise printed notifications for all stages of model interaction (e.g., "Sending Prompt to Code Model...", "Response Received from Code Model...").  
   - Add a configuration page with a non-editable “Prompting” section showing the operation mode, enhancements (e.g., Web-Search, VectorStore), and the last prompt sent.  
   - Remove some of the Debug in terminal, selectively.
4. **Enhanced Notation Modes:**  
   - Introduce “Chat-Notate” and “Chat-Notate-Uncensored” modes to process uploaded PDFs into both detailed and concise summaries.  
   - Store summaries in `.\\data\\notation` and provide a Notation Library menu in the UI for managing these notations.  
   - Disable the THINK phase in these modes to ensure practical, notation-driven conversations.  
5. **Agentic and Enhanced Features:**  
   - Integrate vision capabilities for image recognition on web searches and convert images into contextually relevant thumbnails (with full images accessible via pop-up).  
   - Add voice processing features (Text-to-Voice and Voice-to-Text) using PyWin32 and Whisper.
   - Ensure that attached files can be written to.  
6. **Mode-Specific Integrations and Persistent Sessions:**  
   - Support various modes (Chatbot, Advanced Chatbot, Coder, Agentic Coder, RPG Text, RPG Text + Images) with tailored features as outlined in the design table.  
   - Introduce persistent modes (Chat-Persistent and RPG-Persistent) to enable ongoing sessions with stored states, allowing persistent interactions (e.g., a consistent AI character like a counsellor).  

## Credits
- [Grok3Beta](https://x.com/i/grok) - For much of the complete updated functions that I implemented.
- [Deepseek R1/3](https://www.deepseek.com/) - For re-attempting the things Grok3Beta was having difficulty with.
- [Claude_Sonnet](https://claude.ai) - For a smaller amount of the work on this project, useually when there was issue with the other 2 above.
- [Perplexity](https://www.perplexity.ai) - For research to look at more sites than normal, also notice its agentic now.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

