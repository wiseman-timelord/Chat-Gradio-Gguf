# Chat-Gradio-Gguf
Status: Alpha 

## Description
A high-performance chat interface optimized for GGUF models, designed for efficiency and usability. The project is tailored to my specific needs, ensuring a streamlined and non-bloated experience. With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool eliminates the need for online chatbots while providing local, uncensored, and efficient inference. The interface is designed to evolve with additional features that enhance productivity and usability.

## Features
- **Uncensored Efficiency** - Optimized for Gguf models, automatic num layers on GPU.
- **GPU-First Design** - CUDA 11.7/12.4 focus with NVIDIA 10-series+ compatibility
- **Research-Grade Tools** - Integrated RAG, web search, and cognitive visualization
- **Virtual Environment Support** - Isolated Python dependencies for clean installations
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
- CUDA installer with version selection.  
- Directory structure setup.  
- Basic Gradio interface.  
- Configuration management.  
- Virtual environment integration.  
- Support for **Lamarckvergence-14B** and **Nxcode-CQ-7B** models.  
- Message history management with session labeling.  
### Outstanding  
- **Token Streaming**: Real-time token generation for smoother interactions.  
- **Code Block Formatting**: Automatically format code blocks marked with ` ``` `.  
- **FAISS Vector Storage**: Implement vector storage for enhanced RAG capabilities.  
- **Contextual Injection System**: Dynamically inject context into prompts.  
- **Layer-wise GPU Offloading**: Automatically calculate and allocate layers based on available VRAM.  
- **Session Summarization**: After the first response, summarize the session and generate a 3-word label for the history tab.  
- **Model Selection Dropdown**: Add a dropdown in the Gradio interface to select and switch between models.  
- **Configuration Page Enhancements**: Add dropdowns for settings like temperature (`0`, `0.25`, `0.5`, `0.75`, `1`).  

### Notation
- we are using the calculation `(ModelFileSize * 1.25) / NumLayers = LayerSize` then `TotalVRam / LayerSize = NumLayersOnGpu`, then convert that to whole number, and then load that number of layers of the model to the gpu with the load model command.

## Requirements
- Windows 10/11 - Its a Windows program, it will be also compatible with linux later.
- NVIDIA GPU - CUDA, 11.7 or 12.4, compatible.
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

