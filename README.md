# Chat-Gradio-Gguf
```
=======================================================================================================================
"                                  _________           ________          ________                                     "
"                                  \_   ___ \         /  _____/         /  _____/                                     "
"                                  /    \  \/  ______/   \  ___  ______/   \  ___                                     "
"                                  \     \____/_____/\    \_\  \/_____/\    \_\  \                                    "
"                                   \______  /        \______  /        \______  /                                    "
"                                          \/                \/                \/                                     "
-----------------------------------------------------------------------------------------------------------------------
```
Status: Beta - Testing and bugfixing.

## Description
A high-quality chat interface with 3 auto-detected modes of, operation and interface, for, Codeing, Rpg Simulator, Enhanced Chat, for GGUF models on windows 10 with any GPU.  With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool providing local, uncensored, and inference with features that enhance productivity and usability, even a comparable interface, found on premium AI services, or a far in that direction as Gradio will allow. The configuration is without options reported to make no difference on most models, ensuring a comprehensive yet streamlined experience. Capable of all things possible through simple scripts and awesome libraries and modern GGUF LLMs.

### Features
- **Mode Detection **: 3 modes of operation, `code`, `rpg`, `chat`, with, `Uncensored` and `reasoning`, enhancement, and with relating dynamic, prompts and interface. 
- **GPU Support**: Compatible with AMD/NVIDIA/Intel GPUs via Vulkan/Kompute/Cuda/Avx2, with GPU selection dropdown.
- **Research-Grade Tools**: Includes RAG, web search, chunking, summarization, and code formatting.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Simplified File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **FAISS Vector Database**: Stores numerical vectors, and retrieves based on proximity in meaning, enabling pulling context from documents.
- **Session History Rotation**: 11 Session History slots on rotation, enabling load recent sessions, while not using excessive space. 

### Preview
- The "Conversation" page, some elements are dynamic now...
![preview_image](media/conversation_page.jpg)

- The, refined and improved, "Configuration" page...
![preview_image](media/configuration_page.jpg)

- The Terminal Display...
```
=======================================================================================================================
    Chat-Gradio-Gguf: Launcher
=======================================================================================================================

Starting Chat-Gradio-Gguf...
Starting `launcher` Imports.
`launcher` Imports Complete.
Starting `launcher.main`.
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.















```

## Requirements
- Windows 10/11 - Its a Windows program, it may be linux compatible later (not now).
- Llama.Cpp - Options here for, Avx2, Vulkan, Kompute, Cuda 11, Cuda 12.
- Python => 3.8 - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.
- Llm Model - You will need a Large Language Model in GGUF format, See below

### Instructions
1. Download a "Release" version, when its available, and unpack to a sensible directory, like, `C:\Program_Filez\Chat-Gradio-Gguf` or `C:\Programs\Chat-Gradio-Gguf`. 
2. Right click the file `Chat-Gradio-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load.
3. Select `2` from the Batch Menu, to begin installation.
4. you will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware.
5. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder.
6. After the install completes, check for any install issues, you may need to install again if there are.
7. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Chat-Gradio-Gguf`.
8. You will be greeted with the conversation page, but you will first be going to the configuration page.
9. On the `Configuration` page you would configure appropriately, its all straight forwards.
10. Go back to the `Conversation` page and begin interactions, ensuring to notice features available.

### Notation
- Tabs on left of `Chat` page; "Start New Session" at top, 10-session limit.
- Auto-labeled sessions (e.g., "Query about worms") stored in `.\data\history\*`.
- VRAM dropdown, 1GB to 32GB in steps, this should be your FREE ram available on the selected card.
- We use `(ModelFileSize * 1.1875) / NumLayers = LayerSize`, then `TotalVRam / LayerSize = NumLayersOnGpu`.
- Most GGUF text models will work, keep in mind the applicable keywords shown in `Model Label Keywords` section.
- If you like using LLMs on your own hardware, then you will probably like my program [DownLord](https://github.com/wiseman-timelord/DownLord).

### Model label/name Keywords...
1. Keywords for Operation mode...
- `Coding` keywords - "code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell".
- `RPG Game` keywords - "nsfw", "adult", "mature", "explicit", "rp", "roleplay".
- `Chat` keywords - none of the above.
2. Keywords for Enhancements...
- `UnCensored` keywords - "uncensored", "unfiltered", "unbiased", "unlocked".
- `reasoning` keywords - "reason", "r1", "think".

# Models
You will of course need to have a `*.Gguf` model for anything to work, here are the models used to test the program.. 
1. For ~8B models (Primary/Quality).
- [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) - Best <8b Model on General leaderboard, and ~500 overall.
- Choice between, Llama [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) and Qwen [DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF) , versions of R1 - Uncensored <8GB, Chat and Reasoning.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best on Big code Leaderboard for Python, for Coder.
- [Ninja-v1-NSFW-RP-GGUF](https://huggingface.co/mradermacher/Ninja-v1-NSFW-RP-GGUF) - Most downloaded RP NSFW on huggingface at the time.
2. For <4B models (Secondary/Fast).
- [Llama-3.2-3B-Instruct-uncensored-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF) - untested.
- [DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF) - Uncensored Reasoning.

## Development
Complete Gen 1 of Chat-Gradio-Gguf...
- had to work on this section in `interface.py`, go fully over the keywords system, check it, also check the logic and sanity, ensure its all correct...
```
def determine_operation_mode(quality_model):
    if quality_model == "Select_a_model...":
        return "Select models to enable mode detection."
    settings = get_model_settings(quality_model)
    category = settings["category"]
    if category == "code":
        return "Code"
    elif category == "rpg":  # Changed from "nsfw"
        if "rp" in quality_model.lower() or "roleplay" in quality_model.lower():
            return "Rpg"  # For roleplay-specific models
    elif category == "uncensored":
        return "Chat"
    return "Chat"
```
- go over the full logic and sanity of the vector store, it was printing something like...
```
Vectorstore not found for mode: chat
No vectorstore found for mode: chat
```
...when I select a model, not when I load it. It also shouldnt really be doing anything with the model until I click, load model or inspect model.
- instad of dynamically showing, `History Slot Empry` and `Free File Slot`, or whatever, we would have dynamic fly-out menus. There would be the tabs, `Sessions Menu`, `Attachement Menu`, `Roleplay Menu`. The `Session Menu` button, will make visible, 1 `Start New Session` button and `9 history slot` and 1 `Hide History`, buttons. The `Attachments Menu` will make visible, 1 `Remove Attachments` and 8 `File Slot`buttons and 1 `Hide Attatchments` button, regardless of mode. The `Roleplay Menu`button, is only visible if mode is "RPG", clicking on it will make visible the parameters for roleplay, ie, `human_name: ` and `human_role: `, and also `Hide Roleplay` button, this will solve the issue with space on the right column of the "Chat" menu.

## Far Development.
- the next generation of the program will be 6 modes and dual model like this...
```
| **Mode**             | **Models**                     | **Features**                                      | **Implementation Notes**                                                                 |
|-----------------------|--------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------|
| Chatbot              | Single chat model (e.g., DeepSeek-V3) | TOT reasoning, web research                     | Use LangChain for TOT (e.g., recursive prompt splitting), SerpAPI for web queries       |
| Advanced Chatbot     | Fast + Quality chat models    | TOT on Fast, AUTO/FAST/QUALITY switch, web on Fast | Quantize Fast model (e.g., 4-bit), add Gradio slider for mode, cache Quality outputs   |
| Coder                | Single code model (e.g., DeepSeek-Coder) | Syntax, formatting, completion                | Leverage Pygments for highlighting, integrate VSCode-like keybinds                     |
| Agentic Coder        | Code + text models            | Intent → code → review loop                    | Chain via LangChain agents, store intermediate states in .\data\                       |
| RPG Text             | Single RPG model              | Narrative, character tracking, uncensored       | Parse entities with regex, save JSON states in .\data\history\                         |
| RPG Text + Images    | RPG + image model (Flux.1-schnell) | Text RPG + scene images                      | Trigger Flux via llama.cpp image fork, display in Gradio gallery (256x256 max)         |
```
- Introduction of `Chat-Gradio-Gguf.sh` file and modifications of scripts, to enable, Linux AND Windows, support. 
- Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.
- Color themes. Deepseek R1 AND Grok3Beta, found this extremely difficult, Deepseek was better. It needs to be re-attempted later, possibly with GPT4o1. Make it all shades of grey, then allow people to choose the primary button color in configuration, or something simple like that.

## Credits
- [X](https://x.com/) - [Grok](https://x.com/i/grok), at the time Grok3Beta. For much of the complete updated functions that I implemented.
- [Deepseek](https://www.deepseek.com/), at the time, 3 and R1. For re-attempting the things Grok3Beta was having difficulty with.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

