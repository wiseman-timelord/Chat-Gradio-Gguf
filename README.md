# Chat-Gradio-Gguf
Status: Alpha 

## Description
A high-performance chat interface optimized for GGUF models, designed for efficiency and usability. The project is tailored to my specific needs, ensuring a streamlined and non-bloated experience. With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool eliminates the need for online chatbots while providing local, uncensored, and efficient inference. The interface is designed to evolve with additional features that enhance productivity and usability. The main concept is, to download a DeepSeek R1 model, and use it without the restrictions found on the website, and with a comparable or better interface, or at least as good as it gets before the project becomes too large.

## Features
- **Uncensored Efficiency** - Optimized for Gguf models, automatic num layers on GPU.
- **GPU-First Design** - Made for AMD Non-Rocm cards, but will be compatible with nVidia/others through Vulkan/Kompute.
- **Research-Grade Tools** - Integrated RAG, web search, chunking, summarization.
- **Virtual Environment** - Isolated Python installations to a `.\venv` folder.
- **Simplified File Support** - Read and utilize `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files
- **Configurable Context Window** - Set `n_ctx` to 8192, 16384, 24576, or 32768
- **Enhanced Interface Controls** - Load/Unload model, restart session, and shutdown program with ease

## Installation
**Install Process**  
1. Select CUDA version during installation  
2. Automatic virtual environment setup (`.venv`)  
3. Python dependency installation (Gradio, LangChain, llama-cpp-python)  
4. llama.cpp binary download and extraction  
5. Configuration file generation  

## Development Roadmap
### Completed  
- Llama.cpp installer with variant selection.  
- Directory structure setup.  
- Basic Gradio interface.  
- Configuration management.  
- Virtual environment integration.  
- Support `Lamarckvergence-14B-GGUF` model (current best model under 15b).  
- Message history management with session labeling.  
### Outstanding  
- **Token Streaming**: Real-time token generation for smoother interactions.  
- **Code Block Formatting**: Automatically format code blocks marked with ` ``` `.  
- **FAISS Vector Storage**: Implement vector storage for enhanced RAG capabilities.  
- **Contextual Injection System**: Dynamically inject context into prompts.  
- **GPU Selection**: The user may have more than 1 card, so somehow we need to be able to detect the cards, and offer the user the option to select beteen them from auto-populated dropdown box.
- **Layer-wise GPU Offloading**: Automatically calculate and allocate layers based on available VRAM. (we must make the user specify how much VRam the card has in settings, as AMD non-Rocm cards do not expose this, default to 8). 
- **Session Summarization**: After the first response from the AI, the model would immediately be used in a hidden (non-session aware) interaction with a special prompt, to summarize the nature of the session in 3-words, based on the, input and response, that will become the `Three word label` for the new session in the history tab, and then it will require to switch back to normal operation, with awareness of the session.  
- **Model Selection Dropdown**: Add a dropdown in the Gradio interface to select and switch between models.  
- **Configuration Page Enhancements**: Add dropdowns for settings like temperature (`0`, `0.25`, `0.5`, `0.75`, `1`).  

### Notation
- we are using the calculation `(ModelFileSize * 1.25) / NumLayers = LayerSize` then `TotalVRam / LayerSize = NumLayersOnGpu`, then convert that to whole number, and then load that number of layers of the model to the gpu with the load model command.

## Requirements
- Windows 10/11 - Its a Windows program, it will be also compatible with linux later.
- Llama.Cpp - Compatible here with, Avx2, Vulkan, Kompute (Experimental Vulkan+).
- Python => 3.8 - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.

## Links
- [Lamarckvergence-14B-i1-Gguf](https://huggingface.co/mradermacher/Lamarckvergence-14B-i1-GGUF) - Best Small Chat Model in Gguf format.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best Python code mode in GGUF format.

## Ideas - Not to be implemented currently.
- **Dual-Model Support**: Load two models simultaneously—one for chat and one for code generation. The chat model would act as an agent, sending prompts to the code model and integrating responses into the chat log.  
  - Example Workflow: 1. Chat model sends a prompt to the code model. 2. Code model generates code and sends it back. 3. Chat model reviews and integrates the response.  
  - Logging: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

