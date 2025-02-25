# Chat-Gradio-Gguf
Status: Alpha - We are now experimenting using Grok3 for the creation of this program, so far, so good.

## Description
A high-performance chat interface optimized for GGUF models, designed for efficiency and usability. The project is tailored to my specific needs, ensuring a streamlined and non-bloated experience. With the latest advancements in GGUF models, such as the models found in the `Links` section. This tool eliminates the need for online chatbots while providing local, uncensored, and efficient inference. The interface is designed to evolve with additional features that enhance productivity and usability. The main concept is, to download the best smaller models on HuggingFace, and use them, without the restrictions and with comparable interface, found on premium AI services.

## Features
- **Uncensored Efficiency**: Optimized for GGUF models like `Lamarckvergence-14B-GGUF`, auto-calculates GPU layers.
- **GPU Support**: Compatible with AMD/NVIDIA/Intel GPUs via Vulkan/Kompute, with GPU selection dropdown.
- **Research-Grade Tools**: Includes RAG with FAISS, web search, chunking, summarization, and code formatting.
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Simplified File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml` files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Token Streaming**: Real-time token generation for seamless chat interactions.
- **Model Selection**: Dropdown lists GGUF models from `.\models\*.gguf` for easy switching.

## Requirements
- Windows 10/11 - Its a Windows program, it may be linux compatible later (not now).
- Llama.Cpp - Compatible here with, Avx2, Vulkan, Kompute (Experimental Vulkan+).
- Python => 3.8 - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.

## Installation
**Install Process**  
1. Select Llama.Cpp version during installation  
2. Automatic virtual environment setup (`.venv`)  
3. Python dependency installation (Gradio, LangChain, llama-cpp-python)  
4. llama.cpp binary download and extraction  
5. Configuration file generation with, llama and vulkan, paths noted.   

### Notation
- Tabs on left of `Chat` page; "Start New Session" at top, 10-session limit.
- Auto-labeled sessions (e.g., "Query about worms") stored in `.\data\history\*`.
- Vulkan/Kompute supports all GPUs; optimized for non-ROCM AMD without extras.
- Loads from `.\models` with VRAM dropdown (1GB to 32GB, default 8GB).
- Settings tab offers temperature (-1 to 1) and VRAM options via dropdowns.
- We use `(ModelFileSize * 1.25) / NumLayers = LayerSize`, then `TotalVRam / LayerSize = NumLayersOnGpu`.
- Result is rounded to a whole number for GPU layer offloading in the load model command.

### Development
- Implement enhancements based on research of models stated below.
- Implement enhanced thinking section, for the models that have it, soes not to be using it if the model is not a thinking model, but if it is because it is having a hotword related to the models we found to specifically be thinking models, such as `R1`, then have some kind of `Thinking:`, and on the new line, for each line of thought we would represent this with an additional `█` chatacter, until it finishes thinking, at which point, it would on the new line, print `Thought for #.#s.`, then output the response `Response:\n Example response text`, keeps things simple. 

## Links
- [Lamarckvergence-14B-Gguf](https://huggingface.co/mradermacher/Lamarckvergence-14B-GGUF) - Best Small Chat Model in Gguf format, filename `Lamarckvergence-14B.Q6_K_M.gguf`.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best Python code model in GGUF format, filename `Nxcode-CQ-7B-orpo.Q6_K.gguf`.
- [DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner-GGUF) - Possibly good Uncensored Small Chat model in GGUF format, filename `DeepSeek-R1-Distill-Qwen-14B-Uncensored-Reasoner.Q6_K.gguf`.
- [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) - Interesting Uncensored <8GB Chat model in GGUF format, filename `DeepSeek-R1-Distill-Llama-8B-Uncensored.Q6_K.gguf`.
- [Deepseek-R1-Distill-NSFW-RPv1-GGUF](https://huggingface.co/mradermacher/Deepseek-R1-Distill-NSFW-RPv1-GGUF) - Somewhat ok fast Nsfw Chat model in GGUF format, filename `Deepseek-R1-Distill-NSFW-RPv1.Q6_K.gguf`.
- [Llama-3.2-3b-NSFW_Aesir_Uncensored-GGUF](https://huggingface.co/Novaciano/Llama-3.2-3b-NSFW_Aesir_Uncensored-GGUF) - Good fast Nsfw Chat model in GGUF format, filename `Llama-3.2-3b-NSFW_Aesir_Uncensored.gguf`.


## Ideas - Not to be implemented currently.
- **Dual-Model Support**: Load two models simultaneously—one for chat and one for code generation. The chat model would act as an agent, sending prompts to the code model and integrating responses into the chat log.  
- Example Workflow: 1. Chat model requries code, so it sends a prompt to the code model with requirements and appropriate snippets. 2. Code model generates/updates code and sends it back. 3. Chat model reviews and integrates the response to the appropriate script(s), and saves the relevant script(s). This would also require setting a second model for code, and possibly some kind of, `Chat, Chat-Code, Code`, type of switch in the chat page, to select mode of operation, and it would detect changes in the switch, and load appropriate models. `Chat` would be normal chat to chat model, `Chat-Code` would be using the chat model to form improved prompts to send to the code model, while for the chat model to then be acting appropriately, essentially the code model would become an agent under the control of the code model, and the `Code` mode would be chatting with only the code model. 
- Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

