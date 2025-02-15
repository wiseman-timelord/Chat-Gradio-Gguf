# Chat-Gradio-Deep
A high-performance chat interface for DeepSeek's R2 Distill models, combining GPU efficiency with ChatGPT-like usability.

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
### Phase 1: Core Infrastructure (Completed)
- [X] CUDA installer with version selection
- [X] Directory structure setup
- [X] Basic Gradio interface
- [X] Configuration management
- [X] Virtual environment integration

### Phase 2: Chat Interface (Current Focus)
- [ ] Message history management
- [ ] Token streaming
- [ ] Code block formatting
- [ ] Brainmap visualization prototype

### Phase 3: RAG Integration
- [ ] PDF/DOCX parsing
- [ ] FAISS vector storage
- [ ] Contextual injection system

### Phase 4: Performance Optimization
- [ ] VRAM management
- [ ] 4-bit quantization support
- [ ] Layer-wise GPU offloading

### Phase 5: Deployment Ready
- [ ] Windows installer package
- [ ] Model download helper
- [ ] Security audits

## Key Dependencies
| Component           | Version | Purpose                          |
|---------------------|---------|----------------------------------|
| llama.cpp           | b4722   | GPU-accelerated inference        |
| Gradio              | 4.30.0  | Web interface framework          |
| LangChain           | 0.2.1   | RAG pipeline management          |
| llama-cpp-python    | 0.2.57  | Python bindings for llama.cpp    |
| FAISS               | 1.8.0   | Vector storage and search        |

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

