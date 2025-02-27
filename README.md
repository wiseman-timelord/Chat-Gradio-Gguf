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
Status: Alpha - Early development, but moving along fast.

## Description
A high-quality chat interface optimized for GGUF models, designed for efficiency and usability. The project is tailored to my specific needs, ensuring a streamlined and non-bloated experience. With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool eliminates the need for online chatbots while providing local, uncensored, and efficient inference. The interface is designed to evolve with additional features that enhance productivity and usability. The main concept is, to download the best smaller models on HuggingFace, and use them, without the restrictions and with comparable interface, found on premium AI services.

### Features
- **Uncensored Efficiency**: Optimized for GGUF, auto-calculating layers, dependent on, model details and Free VRam.
- **GPU Support**: Compatible with AMD/NVIDIA/Intel GPUs via Vulkan/Kompute, with GPU selection dropdown.
- **Research-Grade Tools**: Includes RAG with FAISS, web search, chunking, summarization, and code formatting.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Simplified File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Token Streaming**: Real-time token generation for seamless chat interactions.
- **Model Selection**: Dropdown lists GGUF models from `.\models\*.gguf` for easy switching.
- **Auto-Model Handle**: Uses hotwords in model name, to better handle, ctx, temperature, modes, prompts, etc.

### Preview
- The "Conversation" page...
![preview_image](media/conversation_page.jpg)

- The "Configuration" page...
![preview_image](media/configuration_page.jpg)

- The installer works effectively...
```
========================================================================================================================
    Chat-Gradio-Gguf: Installation
========================================================================================================================

Installing Chat-Gradio-Gguf...
Found Vulkan SDK at C:\Program_Filez\VulkanSDK\1.4.304.1 with version 1.4.304.1
Detected Vulkan versions: {'1.4.304.1': WindowsPath('C:/Program_Filez/VulkanSDK/1.4.304.1')}
Confirmed Vulkan SDK 1.4.x version: 1.4.304.1
Found directory: data                                        [OK]
Found directory: files                                       [OK]
Found directory: scripts                                     [OK]
Found directory: models                                      [OK]
Found directory: data/vectorstores                           [OK]
Found directory: data/history                                [OK]
Found directory: data/temp                                   [OK]
Replacing Virtual Environment.                               [OK]
Installing Python Dependencies...                            [OK]
Requirement already satisfied: pip in c:\program_filez\chat-gradio-gguf\.venv\lib\site-packages (22.3)
Collecting pip
  Using cached pip-25.0.1-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 22.3
    Uninstalling pip-22.3:
      Successfully uninstalled pip-22.3
Successfully installed pip-25.0.1
Pip upgraded successfully                                    [OK]
Installing dependencies with custom wheel index...           [OK]
Dependencies installed in venv                               [OK]
Downloading llama.cpp (GPU/CPU - Kompute)...                 [OK]
100%|█████████████████████████████████████████████████████████████████████████████| 17.5M/17.5M [03:20<00:00, 87.1kB/s]
llama.cpp installed successfully                             [OK]
Configuration file over-written                              [OK]
Chat-Gradio-Gguf installed successfully!                     [OK]
 Press any key for Batch Menu...

```

## Requirements
- Windows 10/11 - Its a Windows program, it may be linux compatible later (not now).
- Llama.Cpp - Compatible here with, Avx2, Vulkan, Kompute (Experimental Vulkan+).
- Python => 3.8 - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.

### Instructions
1. Download a "Release" version, when its available, and unpack to a sensible directory, like, `C:\Program_Filez\Chat-Gradio-Gguf` or `C:\Programs\Chat-Gradio-Gguf`. 
2. Right click the file `Chat-Gradio-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load.
3. Select `2` from the Batch Menu, to begin installation.
4. you will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware.
5. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder.
6. After the install completes, check for any install issues, you may need to install again if there are.
7. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Chat-Gradio-Gguf`.
 
### Notation
- Tabs on left of `Chat` page; "Start New Session" at top, 10-session limit.
- Auto-labeled sessions (e.g., "Query about worms") stored in `.\data\history\*`.
- Vulkan/Kompute supports all GPUs; optimized for non-ROCM AMD without extras.
- You will of course need to have a `*.Gguf` model in `.\models`, in order for anything to work.
- VRAM dropdown, 1GB to 32GB in steps, this should be your FREE ram on the selected card.
- Settings tab offers temperature (-1 to 1) and VRAM options via dropdowns.
- We use `(ModelFileSize * 1.25) / NumLayers = LayerSize`, then `TotalVRam / LayerSize = NumLayersOnGpu`.
- Result is rounded to a whole number for GPU layer offloading in the load model command.

### Links
- [Lamarckvergence-14B-Gguf](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) - Best <15b Model at at the time in Gguf format, filename `Lamarckvergence-14B.Q6_K_M.gguf`.
- [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) - Best <8b Model at the time in Gguf format, filename `qwen2.5-7b-cabs-v0.4.Q6_K.gguf`.  
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best Python code model  at the time in GGUF format, filename `Nxcode-CQ-7B-orpo.Q6_K.gguf`.
- [DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner-GGUF) - Possibly good Uncensored Small Chat model in GGUF format, filename `DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner.Q6_K.gguf`.
- [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) - Interesting Uncensored <8GB Chat model in GGUF format, filename `DeepSeek-R1-Distill-Llama-8B-Uncensored.Q6_K.gguf`.
- [Llama-3.2-3b-NSFW_Aesir_Uncensored-GGUF](https://huggingface.co/Novaciano/Llama-3.2-3b-NSFW_Aesir_Uncensored-GGUF) - Tested and somewhat good, fast Nsfw Chat model in GGUF format, filename `Llama-3.2-3b-NSFW_Aesir_Uncensored.gguf`.

## Development
- Testing and bugfixing, required for all features of main program.
- Possibly we need options to turn off advanced features like RAG, idk, needs assessment, would no rag disable attach file options? In which case the button for attach files would be shown dynamically. 
- The session history column needs to be 2/3 its current width.
- It needs a `n_batch` dropdown in configuration in the row of options starting with temperature, and it should be after `Context Window`, labelled `Batch Size`, withe the options `128, 256, 512, 1024, 2048, 4096`, defaulting to `1024`, this will require again, line with + options in temporary, and a line in json for the variable, and possibly an update to the load/save/update config functions, as well as how it interacts with llama in the argument.
- Is the status text in the correct location, should it be under the buttons, make 2 images and compare side by side.
- The `User Input` box should automatically expand to up to 5 lines to compensate for text input amount, while symultaneously the `Session Log` box would relatively shorten by up to 5 lines, and when the text input box goes over 5 lines of input, it should automatically gain a scrolling feature. Simlarly, if the text for the `Session Log` is larger than the box size, then it should also automatically add a slider, to allow the user to scroll through the history in the session. Like regular chatbots work.  
- with regards to rag settings, possibly things like, `chunk_size` and `chunk_overlap`, should be calculated based on a percentage of the `n_ctx`, possibly, `n_ctx / 2 = chunk_size` and `chunk_overlap / 16 = chunk_overlap`, this way we could reduce by 2 keys in the Json. The re-calculaton would be done at the optimal point, im guessing that would be if the n_ctx changes or model is loaded/reloaded.
- there is a key for `max_docs`, this should be next to `max_session_history` option on `configuration` page, and have options, `2, 4, 6, 8` defaulting to `4`.
- Ongoing improvement of Gradio Interface required, until it "looks right". 
- It would be nice to be able to set the model folder, again, key in, temporary and json, plus relevant updates in applicable functons.
- I want to rename `.\data\config.json` to `.\data\persistent.json`, simple enough.

### Ideas - Not to be implemented currently.
- Introduction of `Chat-Gradio-Gguf.sh` file and modifications of scripts, to enable, Linux AND Windows, support. 
- Agentic workflows, potentially using purpose built fine tuned models, for, thinking, interaction, or code. 
- There is also an idea of the `Performance` model and the `Quality` model, where the user could switch between, albeit this could also be, fast for simple tasks like creating the title of the session, and slow for interaction, or the likes.
- Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

