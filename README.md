# Chat-Gradio-Gguf
Status: Alpha - We are now experimenting using Grok3 for the creation of this program, so far, so good.

## Description
A high-performance chat interface optimized for GGUF models, designed for efficiency and usability. The project is tailored to my specific needs, ensuring a streamlined and non-bloated experience. With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool eliminates the need for online chatbots while providing local, uncensored, and efficient inference. The interface is designed to evolve with additional features that enhance productivity and usability. The main concept is, to download the best smaller models on HuggingFace, and use them, without the restrictions and with comparable interface, found on premium AI services.

## Features
- **Uncensored Efficiency** - Optimized for Gguf models, automatic calculation of num layers on GPU.
- **GPU-First Design** - Will be compatible with Amd/nVidia/intel through Vulkan/Kompute.
- **Research-Grade Tools** - Integrated RAG, web search, chunking, summarization.
- **Virtual Environment** - Isolated Python installations to a `.\venv` folder.
- **Simplified File Support** - Read and utilize `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files
- **Configurable Context Window** - Set `n_ctx` to 8192, 16384, 24576, or 32768
- **Enhanced Interface Controls** - Load/Unload model, restart session, and shutdown program with ease

## Installation
**Install Process**  
1. Select Llama.Cpp version during installation  
2. Automatic virtual environment setup (`.venv`)  
3. Python dependency installation (Gradio, LangChain, llama-cpp-python)  
4. llama.cpp binary download and extraction  
5. Configuration file generation with, llama and vulkan, paths noted.   

## Development
All this requires to be implemented, if not already present...  
- Llama.cpp installer with variant selection.  
- Directory structure setup.  
- Basic Gradio interface.  
- Configuration management.  
- Virtual environment integration.  
- Support chat models such as `Lamarckvergence-14B-GGUF` (current best model under 15b).  
- Message history management with sessions archived in auto-labelled tabs on left on gradio `Chat` page.  
- **Token Streaming**: Real-time token generation for smoother interactions.  
- **Code Block Formatting**: Automatically format code blocks marked with ` ``` `.  
- **FAISS Vector Storage**: Implement vector storage for enhanced RAG capabilities.  
- **Contextual Injection System**: Dynamically inject context into prompts.  
- **GPU Selection**: The user may have more than 1 card, so somehow we need to be able to detect the cards, and offer the user the option to select beteen them from auto-populated dropdown box.
- **Layer-wise GPU Offloading**: Automatically calculate and allocate layers based on available VRAM. (we must make the user specify how much free VRam the card has in settings with a dropdown box with options `1GB, 2GB, 3GB, 4GB, 6GB, 8GB, 10GB, 12GB, 16GB, 20GB, 24GB, 32GB`, as some cards, such as AMD non-Rocm cards, do not expose total VRam size, default to `8GB`. 
- **Session Summarization**: After the first response from the AI, the model would immediately be used in a hidden (non-session aware) interaction with a special prompt, to summarize the nature of the session in 3-words, based on the, input and response, that will become the `Three word label` for the new session in the session history tabs on the left of the `Chat` page in the gradio interface, and then it will require to switch back to normal operation, with awareness of the session and the label set on the current session's tabs. The top tab should always be `Start New Session`, and the user will click on it to create a new session, while when there is a label for the session, it will be created under the top tab. as with regular modern chat interfaces, the user should be able to switch between the tabs, which will require some method of archiving the sessions, however, unlike modern chat interfaces, I want the history of sessions, to be limited to the number of provisionally 10 sessions, to keep things tidy, when there are 11 sessions archived, then the oldest one will automatically be deleted. History will be in `.\data\history\*` with an individual file for each one, therein we need to know which ones are newer in order to display them with the newest first next to the fake tab used to `Start New Session`, there would need to be a maintenance service after adding a new session, however that would be done best. 
- **Model Selection Dropdown**: Add a dropdown in the Gradio interface to select and switch between models, it should automatically populate the list from the GGUF models available here `.\models\*.gguf`, and yes, the models are intended to go directly into `.\models`, not `.\data\models` or some other folder. 
- **Configuration Page Enhancements**: Add dropdowns for settings like temperature (`-1`, `-0.75`, `-0.5`, `-0.25`, `0`, `0.25`, `0.5`, `0.75`, `1`).  

### Notation
- we are using the calculation `(ModelFileSize * 1.25) / NumLayers = LayerSize` then `TotalVRam / LayerSize = NumLayersOnGpu`, then convert that to whole number, and then load that number of layers of the model to the gpu with the load model command.

## Requirements
- Windows 10/11 - Its a Windows program, it will be also compatible with linux later.
- Llama.Cpp - Compatible here with, Avx2, Vulkan, Kompute (Experimental Vulkan+).
- Python => 3.8 - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.

## Links
- [Lamarckvergence-14B-Gguf](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) - Best Small Chat Model in Gguf format.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best Python code model in GGUF format.

## Ideas - Not to be implemented currently.
- **Dual-Model Support**: Load two models simultaneously—one for chat and one for code generation. The chat model would act as an agent, sending prompts to the code model and integrating responses into the chat log.  
  - Example Workflow: 1. Chat model requries code, so it sends a prompt to the code model with requirements and appropriate snippets. 2. Code model generates/updates code and sends it back. 3. Chat model reviews and integrates the response to the appropriate script(s), and saves the relevant script(s). This would also require setting a second model for code, and possibly some kind of, `Chat, Chat-Code, Code`, type of switch in the chat page, to select mode of operation, and it would detect changes in the switch, and load appropriate models. `Chat` would be normal chat to chat model, `Chat-Code` would be using the chat model to form improved prompts to send to the code model, while for the chat model to then be acting appropriately, essentially the code model would become an agent under the control of the code model, and the `Code` mode would be chatting with only the code model. 
  - Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

