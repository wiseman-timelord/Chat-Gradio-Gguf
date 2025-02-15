# Chat-Gradio-Deep
Status: Alpha 

## Description
A high-performance chat interface optimized for whatever I find to be the current best model, that will be mentioned below in the links section, combining GPU efficiency with ChatGPT-like usability. It will be designed specifically for my usage, and necessity for features, so it will be streamlined, and non-bloated. I now have confidence that the following generations of models are going to be Epic, and remove any need for me to be using online chatbots, other than speed of inference, but still, after I have made the thing, I will be looking to enhance the thing, and who knows, some kind of amazing additional features, after I get it up to speed, that will totally make my life easier, and I am totally going to borrow ideas, just as they do, to keep up with each other.

## Features
- **Uncensored Efficiency** - Optimized for Gguf models, automatic num layers on GPU.
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
- after the first response to the user, the model should have an additonal response with a special prompt, where it is asked to summarize the session, and then create a 3 word description, that it will use for the label on the history tab. It should not add the, prompt or response, for figuring out the label to the session, other than on the session label.
- drop down box in configuration page in gradio interface, to select model to use, with a refresh button next to it.
- Other things in config page of gradio interface will need drop-down boxes where possible, for example, sensible options for temperature would be, `0`, `0.25`, `0.5`, `0.75`, `1`.

## Potential Development - Not to be implemented currently.
- We could load 2 models, one for chat and thinking, and one for code production, load both at the same time, with equal number of layers from each model, whatever will fit, but both models have some layers on the gpu, the rest in system ram. the best small big code model on huggingface is `Nxcode-CQ-7B-orpo`, specifically the file `Nxcode-CQ-7B-orpo-Q6_K.gguf`, its the best at python, so better than other 70b or 32b models, and as a result I should still have plenty of system ram free. The idea is the chat model would be made aware of this other model, possibly it would be done as an agent, and the chat model would be filling out prompts of what it wants created, and sending that to the code model, then the code model would respond to the chat model, and there would be a note in the chatlog in the session `Sending Prompt to Code Model...` and `...Generating Code for Chat Model...` and `...Response Received from Code Model`, then begin output, where it will have assessed and checked what it received, or re-prompt appropriately, or something of the likes.

## Requirements
- Python Libraries - Gradio, LangChain, llama-cpp-python, FAISS.
- Pre-Compiled Binaries - llama.cpp (b4722).

## Links
- (Lamarckvergence-14B-i1-Gguf)[https://huggingface.co/mradermacher/Lamarckvergence-14B-i1-GGUF] - Best Chat Modelin Gguf format.
- (Nxcode-CQ-7B-orpol-Gguf)[https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF] - Best code mode in GGUF format.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

