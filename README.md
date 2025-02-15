# Chat-Gradio-Deep
Status: Alpha 

## Description
A high-performance chat interface for DeepSeek's R2 Distill models, combining GPU efficiency with ChatGPT-like usability. It will be designed specifically for my programming projects, as I now have confidence that the following models from deepseek, are going to be Epic, and remove any need for me to be using online chatbots, other than speed of inference, but still, after I have made the thing, I will be looking to enhance the thing, and who knows, some kind of amazing additional features, after I get it up to speed, that will totally make my life easier, and I am totally going to borrow ideas, just as they do, to keep up with each other.

## Features
- **Uncensored Efficiency** - Optimized for DeepSeek's R2 Distill models (8B/32B/70B GGUF)
- **GPU-First Design** - CUDA 11.7/12.4 focus with NVIDIA 10-series+ compatibility
- **Research-Grade Tools** - Integrated RAG, web search, and cognitive visualization
- **Virtual Environment Support** - Isolated Python dependencies for clean installations
- **Simplified File Support** - Read and utilize `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files
- **Configurable Context Window** - Set `n_ctx` to 8192, 16384, 24576, or 32768
- **Enhanced Interface Controls** - Load/Unload model, restart session, and shutdown program with ease

## Installation
**System Requirements**  
- NVIDIA GPU with CUDA 11.7 or 12.4
- Windows 10/11
- Python 3.8 or higher

**Install Process**  
1. Select CUDA version during installation  
2. Automatic virtual environment setup (`.venv`)  
3. Python dependency installation (Gradio, LangChain, llama-cpp-python)  
4. llama.cpp binary download and extraction  
5. Configuration file generation  

## Development Roadmap
- DONE:
- CUDA installer with version selection
- Directory structure setup
- Basic Gradio interface
- Configuration management
- Virtual environment integration
- OUTSTANDING:
- Message history management (I dont understand, is this for saving previous sessions, or for within a session? It would be nice to have another column on the left, that saved a history for each session in a folder such as `.\history\######` where `#` is a number in serial, but in the file it should have a title at the top and any additional info required, and just save them as `.txt` files, then we can remove `.\logs` so long as errors are printed to the terminal, we dont need to log terminal activity.
- Token streaming
- Code block formatting triggered by sections marked with "```" and "```", being printed.
- Input file formats accepted, `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`.
- FAISS vector storage
- Contextual injection system
- 4-bit quantization support (though I dont understand why this is an item in the list, surely llama.cpp binaries just load Q_4, Q_5, Q_6, all the same way? I dont only want to use Q_4, maybe I dont understand).
- Layer-wise GPU offloading (assess free vram, assess size of model, assess how many layers the model has, determine layer size by using the calculations `(ModelSize * 1.25) / NumLayers) = SizePerLayer` and `VRamSize / SizePerLayer = NumLayersFitVRam`, then we would use whatever whole number for `NumLayersFitVRam` as the number of layers to put on VRam. 

## Requirements
- Python Libraries - Gradio, LangChain, llama-cpp-python, FAISS.
- Pre-Compiled Binaries - llama.cpp (b4722).

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

