# Chat-Gradio-Deep
A high-performance chat interface for DeepSeek's R2 Distill models, combining GPU efficiency with ChatGPT-like usability.

![Project Schematic](https://via.placeholder.com/800x400.png?text=Chat-Gradio-Deep+Interface+Preview)

## Why This Project?
- **Uncensored Efficiency** - Optimized for DeepSeek's R2 Distill models (8B/32B/70B GGUF)
- **GPU-First Design** - CUDA 11.7/12.4 focus with NVIDIA 10-series+ compatibility
- **Research-Grade Tools** - Integrated RAG, web search, and cognitive visualization

## Core Features
✅ **Completed in Phase 1**  
- CUDA-optimized installer with version selection
- Gradio interface scaffolding
- Configuration system with JSON management
- Directory structure for model/vectorstore management

🚧 **Upcoming Features (Phases 2-5)**  
- Cognitive Process Visualizer (Brainmap)
- File-aware RAG pipeline
- Web search integration
- 4-bit quantized model support
- Real-time token streaming

## Installation
**System Requirements**  
- NVIDIA GPU with cuda 11-12
- Windows 10/11
- CUDA Toolkit 11.7 or 12.4

```bash
git clone https://github.com/yourusername/Chat-Gradio-Deep
cd Chat-Gradio-Deep
python installer.py
```

**Install Process**  
1. Select CUDA version during installation  
2. Automatic model directory setup  
3. Configuration file generation  
4. Dependency installation (Gradio, LangChain, llama.cpp)

## Configuration
`data/configuration.json` structure:
```json
{
  "model_settings": {
    "model_path": "data/models/deepseek-r2-distill.Q4_K_M.gguf",
    "n_gpu_layers": 35,
    "temperature": 0.7,
    "llama_cli_path": "data/llama-cu11.7-bin/bin/llama-cli.exe"
  },
  "ui_settings": {
    "font_size": 14,
    "accent_color": "#4CAF50"
  }
}
```

## Development Roadmap
### Phase 1: Core Infrastructure (Completed)
- [X] CUDA installer with version selection
- [X] Directory structure setup
- [X] Basic Gradio interface
- [X] Configuration management

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
| Component       | Version | Purpose                          |
|-----------------|---------|----------------------------------|
| llama.cpp       | b4722   | GPU-accelerated inference        |
| Gradio          | 4.30.0  | Web interface framework          |
| LangChain       | 0.2.1   | RAG pipeline management          |
| llama-cpp-python| 0.2.57  | Python bindings for llama.cpp    |

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License
It will be a `Wiseman-Timelord's Glorified License`.
