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
Status: Alpha - Mid-development.

## Description
A high-quality chat interface with 4 auto-detected modes of, operation and interface, for, Coder, Rp Simulator, Uncensored Chat, General Chat, for GGUF models on windows 10 with any GPU.  With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool providing local, uncensored, and inference with features that enhance productivity and usability, even a comparable interface, found on premium AI services, or in that direction. The configuration is without options reported to make no difference on most models, ensuring a comprehensive yet streamlined experience. Capable of all things possible through simple scripts and awesome libraries and text based LLM(s).

### Features
- **Uncensored Efficiency**: Optimized for GGUF, auto-calculating layers, dependent on, model details and Free VRam.
- **GPU Support**: Compatible with AMD/NVIDIA/Intel GPUs via Vulkan/Kompute/Cuda/Avx2, with GPU selection dropdown.
- **Research-Grade Tools**: Includes RAG, web search, chunking, summarization, and code formatting.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Simplified File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Token Streaming**: Real-time token generation for seamless chat interactions.
- **Model Selection**: Dropdown lists GGUF models from `.\models\*.gguf` for easy switching.
- **Auto-Model Handle**: Uses keywords in model name, to better handle, ctx, temperature, modes, prompts, etc.
- **FAISS Vector Database**: Stores numerical vectors, and retrieves based on proximity in meaning, enabling pulling context from documents.

### Preview
- The "Conversation" page, still a few things to work out, but mostly there...
![preview_image](media/conversation_page.jpg)

- The "Configuration" page, adding logically, but keeping critical, soes, no bloat...
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

### Model Keywords for Operational Modes
Layers for GPU is auto-detected, there are then keywords on the label of the model...
- `Coding` keywords - "code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell".
- `RPG Game` keywords - "nsfw", "adult", "mature", "explicit", "rp", "roleplay".
- `UnCensored Chat` keywords - "uncensored", "unfiltered", "unbiased", "unlocked".
- `General` keywords - none of the above.

# Models
Most GGUF text models will work, so long as they have the appropriate keywords in the label, which will require time to develop, working on testing with the ones below... 
1. For ~8B models (Primary/Quality).
- [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) - Best <8b Model on General leaderboard, and ~500 overall.
- Choice between, Llama [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) and Qwen [DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF) , versions of R1 - Uncensored <8GB, Chat and Reasoning.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best on Big code Leaderboard for Python, for Coder.
- [Ninja-v1-NSFW-RP-GGUF](https://huggingface.co/mradermacher/Ninja-v1-NSFW-RP-GGUF) - Most downloaded RP NSFW on huggingface at the time.
2. For <4B models (Secondary/Fast).
- [Llama-3.2-3B-Instruct-uncensored-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF) - untested.
- [DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF) - Uncensored Reasoning.

### Notation
- Tabs on left of `Chat` page; "Start New Session" at top, 10-session limit.
- Auto-labeled sessions (e.g., "Query about worms") stored in `.\data\history\*`.
- Vulkan/Kompute supports all GPUs; optimized for non-ROCM AMD without extras.
- You will of course need to have a `*.Gguf` model in `.\models`, in order for anything to work.
- VRAM dropdown, 1GB to 32GB in steps, this should be your FREE ram on the selected card.
- Settings tab offers temperature (-1 to 1) and VRAM options via dropdowns.
- We use `(ModelFileSize * 1.25) / NumLayers = LayerSize`, then `TotalVRam / LayerSize = NumLayersOnGpu`.
- Result is rounded to a whole number for GPU layer offloading in the load model command.
- 9 "History Slots" and "6 File Slots", seemed, like sensible numbers and to fit the display.

## Development
Currently the concept is 4 main modes of operation, `code`, `rpg`, `uncensored`, `chat`, but I want to, re-associate uncensored and produce some theory for 2 model mode operation. 
1. `Uncensored` In the program, this needs to be handled like the Reasoning, that used to be one of the modes but became a feature, and just like the resoning models are handled so as to provide an option to not have the `THINK` phase andy bypass that part, likewise the `uncensored` keywords should be used only automatically to adjust the prompt to unlick the uncensored mode, as should be somewhere in the code already, while for the mode that would have been uncensored to now become merged with general chat mode, as other than the prompt its the same thing. so we need to merge code for, `chat` and `uncensored`, other than the part where it will adapt the prompt for the uncensored operation.
2. I want a load model button, it will do the calculations for the, size and layers, of models configured, for one model load as many layers correctly fit into the VRam onto the GPU, and for 2 models both have an optimal number of layers loaded based on a equal share of the GPU memory, where is one can be completely contained in the model given half of the available vram, then whatever space is left in that half would be added to the free vram in the calculation for the other model's layers.
3. the configuration of the models will affect the configuration of the interface..
- `Fast Chat + Fast Chat` - 3 Way conversation? This would require new modification of interface for conversation, but would activate if you load the exact same model in both slots. 
- `Chat + Fast Chat` - Genearl interaction is done through the Chat Model, while, Fast Chat can enabled for simpler requests and TOT will use fast model also, this where it would be, for example 3K and 6K of same model.
- `RP + Fast Chat` - Fastchat can be use for summarization then use right panel below/above rp details for summarization, also need to work on prompts and use promts from other chat program I made.
- `Reasoning + Code` - Agentic code generation would be the way to do it, it would require figuring out, how I would like it to work, which would involve planning, creation, saving, etc. It cant be too impressive, scripts would end up huge, so start with basic stuff. Possibly could also use resoning model with THINK turned off for scraping web for info to assist projects.
- Single model operation would still be a thing, but whatever tasks would normally be done optimally between modles, would then be done through 1 model, but I need clear definitions for single modle use mode as above, just to clarify. 

### Far Development.
- Testing and Bugfixing - keep testing, while keeping an eye on the terminal for output of, warnings and errors, then find fixes, and a few, refinements or improvements, along the way.
- If there is some small 1B model, we could download and use, to create the session history label for the session, then this would be better, than the user having to wait for whatever more capable model is doing, so as to produce quick labels for session history. preferably a <256MB download, and again, the context size is automatic based on the size of the first, input and response, but if it runs out of context, then we will cut it where it runs out, as for example, if we just have most of the users first input, then this should be good enough to create a unique 3 word label.
- Ongoing improvement of Gradio Interface required, until it "looks right". 
- Testing and bugfixing, required for all features of main program.
- Introduction of `Chat-Gradio-Gguf.sh` file and modifications of scripts, to enable, Linux AND Windows, support. 
- Agentic workflows, potentially using purpose built fine tuned models, for, thinking, interaction, or code. 
- There is also an idea of the `Performance` model and the `Quality` model, where the user could switch between, albeit this could also be, fast for simple tasks like creating the title of the session, and slow for interaction, or the likes.
- Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.

## Credits
- [X](https://x.com/) - [Grok](https://x.com/i/grok), at the time Grok3Beta. For much of the complete updated functions that I implemented.
- [Deepseek](https://www.deepseek.com/), at the time, 3 and R1. For re-attempting the things Grok3Beta was having difficulty with.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

