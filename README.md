# Text-Gradio-Gguf
```
=======================================================================================================================
"                                  ___________      ________          ________                                        "
"                                  \__    ___/     /  _____/         /  _____/                                        "
"                                    |    | ______/   \  ___  ______/   \  ___                                        "
"                                    |    |/_____/\    \_\  \/_____/\    \_\  \                                       "
"                                    |____|        \______  /        \______  /                                       "
"                                                         \/                \/                                        "
-----------------------------------------------------------------------------------------------------------------------
```
Status: Alpha - Testing and bugfixing.

## Description
Intended as a high-quality chat interface with uses include, Codeing, Rpg Game, Chat, programmed towards windows 10 non-WSL, with any Cpu/Gpu on GGUF models. Dynamic modes enabling correct, interface and prompts, for relating theme of sessions, With the latest advancements, and no imposed, limitations or guidelines. This tool providing local, uncensored, and inference with features that enhance productivity and usability, even a comparable interface, found on premium AI services, or as far in that direction as, Gradio and Grok3Beta, will allow. The configuration is without options reported to make no difference on most models, ensuring a comprehensive yet streamlined experience. Capable of all things possible through simple scripts and awesome libraries and modern GGUF LLMs.

### Features
- **Operation Modes **:  "Chat" for general purpose, "Coder" for working on code, "Rpg" for customizable scenario themed interaction.
- **Comprihensive CPU/GPU Support**: CPUs x64-CPU/AVX2/AVX512 and GPUs AMD/nVIDIA/Intel, with dropdown list selection supporting multi CPU/GPU setup.
- **Research-Grade Tools**: Includes RAG, web search, chunking, summarization, TOT, no-THINK, and code formatting, and with file attachments. 
- **Virtual Environment**: Isolated Python setup in `.venv` with `models` and `data` directories.
- **Common File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`, and other common formats of files.
- **Configurable Context Window**: Set `n_ctx` to 8192, 16384, 24576, or 32768 via dropdown.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **FAISS Vector Database**: Stores numerical vectors, and retrieves based on proximity in meaning, enabling pulling context from documents.
- **Highly Customizable UI**: Configurable; 10-20 Session History slots, 2-10 file slots, Session Log 400-550px height, 2-8 Lines of input. 
- **Afterthought Countdown**: If, =>10 lines then 5s or 5-10 lines then 3s or <5 lines then 1s, wait for you to cancel send and return to edit, saving on mis-prompts and waits.

### Preview
- The "Conversation" page, the session rotation, log and input, and the Operation Mode selector with dynamic display elements...
![preview_image](media/conversation_page.jpg)

- The "Configuration" page - for configuration of, models and hardware, and relevant components, as well as ui customization...
![preview_image](media/configuration_page.jpg)


- Wide Llama.Cpp support in the installer, thanks to latest ~128k ai systems, no longer having to streamline such things...
```
========================================================================================================================
    Text-Gradio-Gguf: Install Options
========================================================================================================================



 Select the Llama.Cpp type:

    1. AVX2 - CPU Only - Must be compatible with AVX2

    2. AVX512 - CPU Only - Must be compatible with AVX512

    3. NoAVX - CPU Only - For older CPUs without AVX support

    4. OpenBLAS - CPU Only - Optimized for linear algebra operations

    5. Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU

    6. Kompute - GPU/CPU - Experimental Vulkan with with CPU fallback

    7. CUDA 11.7 - GPU/CPU - For CUDA 11.7 GPUs with CPU fallback

    8. CUDA 12.4 - GPU/CPU - For CUDA 12.4 GPUs with CPU fallback



========================================================================================================================
 Selection; Menu Options = 1-8, Exit Installer = X:

```

## Requirements
- Windows 10/11 - Its a Windows program, it may be linux compatible later (not now).
- Llama.Cpp - Options here for, Avx2, Vulkan, Kompute, Cuda 11, Cuda 12.
- Python => 3.8 (tested on 3.11.0) - Libraries used = Gradio, LangChain, llama-cpp-python, FAISS.
- Llm Model - You will need a Large Language Model in GGUF format, See `Models` section.
- Suitable CPU/GPU - Gpu may be, Main or Compute, with VRam 2-64GB, testing on rx470 in Compute.  

### Instructions
1. Download a "Release" version, when its available, and unpack to a sensible directory, like, `C:\Program_Filez\Text-Gradio-Gguf` or `C:\Programs\Text-Gradio-Gguf`. 
2. Right click the file `Text-Gradio-Gguf.bat`, and `Run as Admin`, the Batch Menu will then load.
3. Select `2` from the Batch Menu, to begin installation.
4. you will be prompted to select a Llama.Cpp version to install, which should be done based on your hardware.
5. After which, the install will begin, wherein Python requirements will install to a `.\venv` folder.
6. After the install completes, check for any install issues, you may need to install again if there are.
7. You will then be returned to the Batch Menu, where you, now and in future, select `1` to run to run `Text-Gradio-Gguf`.
8. You will be greeted with the conversation page, but you will first be going to the configuration page.
9. On the `Configuration` page you would configure appropriately, its all straight forwards.
10. Go back to the `Conversation` page and begin interactions, ensuring to notice features available.

### Notation
- Tabs on left of `Chat` page; "Start New Session" at top, 10-session limit.
- Auto-labeled sessions (e.g., "Query about worms") stored in `.\data\history\*`.
- VRAM dropdown, 1GB to 32GB in steps, this should be your FREE ram available on the selected card.
- We use `(ModelFileSize * 1.1875) / NumLayers = LayerSize`, then `TotalVRam / LayerSize = NumLayersOnGpu`.
- Most GGUF text models will work, keep in mind the applicable keywords shown in `Model Label Keywords` section.
- For downloading large files such as LLM in GGUF format, then typically I would use  [DownLord](https://github.com/wiseman-timelord/DownLord), instead of lfs.

### Model label/name Keywords...
1. Keywords for Operation mode...
- `Coding` keywords - "code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell".
- `RPG Game` keywords - "nsfw", "adult", "mature", "explicit", "rp", "roleplay".
- `Chat` keywords - none of the above.
2. Keywords for Enhancements...
- `UnCensored` keywords - "uncensored", "unfiltered", "unbiased", "unlocked".
- `reasoning` keywords - "reason", "r1", "think".

# Models
You will of course need to have a `*.Gguf` model for anything to work, here are the models used to test the program.. 
1. For ~8B models (Primary/Quality).
- [qwen2.5-7b-cabs-v0.4-GGUF](https://huggingface.co/mradermacher/qwen2.5-7b-cabs-v0.4-GGUF) - Best <8b Model on General leaderboard, and ~500 overall.
- Choice between, Llama [DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Llama-8B-Uncensored-GGUF) and Qwen [DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner-GGUF) , versions of R1 - Uncensored <8GB, Chat and Reasoning.
- [Nxcode-CQ-7B-orpol-Gguf](https://huggingface.co/tensorblock/Nxcode-CQ-7B-orpo-GGUF) - Best on Big code Leaderboard for Python, for Coder.
- [Ninja-v1-NSFW-RP-GGUF](https://huggingface.co/mradermacher/Ninja-v1-NSFW-RP-GGUF) - Most downloaded RP NSFW on huggingface at the time.
2. For <4B models (Secondary/Fast).
- [Llama-3.2-3B-Instruct-uncensored-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF) - untested.
- [DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF](https://huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF) - Uncensored Reasoning.

## Development
With regards to the current version of the program...
1. When given a complicated multi-line prompt, for generation of a storyboard, it suddenly stoped outputting after ~2500 text characters. I set the batch output size to 4096 tokens, this is possibly the token limit for output. Either way, when I stated `please continue` and hit send, it replied with...
```
AI-Chat-Response:
I 'll continue in an unbiased and unc ensored manner . Please provide the context or ask a question , and I 'll respond accordingly . Keep in mind that I 'll give you direct and honest answers , without sugar co ating or filtering out sensitive information .

What 's on your mind ?
```
...so as to have no idea what I was on about. This means the context of the previous interactions is not being injected into the prompt appropriately, or its not being correctly prompted in the second interaction onwards, possibly we would have a second prompt for user interaction, where there is adapted content from other chat mode, for injecting previous context into, would require assessment of, latest most effective, techniqes, as well as any relvant code already present. Additionally investigate what can be done about detection of if the stream has ended prematurely or actually finished output, because if it has not finished output, then it will need to continue, if that can be automated, to a pre-set limit for output, and steram in sections while looking out for end output signature and keeping note of total output. Needs a brainstorm and plan, what is the best solution.
2. I want some better control over prompting, this calls for a new page in the configuration the `Prompting` page, where there will be, some kind of non/editable text indicating the enhancements used and the mode, such as..
```
Operation Mode: 
    Chat

Enhancements: 
    Web-Search, VectorStore

Prompt Contents:
...
```
- a non-editable text box `Last Prompt Sent` displaying the most recent prompt sent to the model, if it also had a section above the prompt, for example...
```

...
```
...then having the raw contents of the prompt sent to the model.

to show if the relevant features were used in that prompt, such as, tot, websearch  were with the prompting options avtive shown above it in some display showing al above the pro 
4. History slots are having typically 1 word labels with 17 chracter date/time, even though I specified for generated label to have 4-5 word description, possibly when the 1st response comes back from the AI, then also again we would generate a label from the contents of now both communications, and update the label on the session with the new completed label. also if there is no response from the AI within a history session, and then the user switches to a different session or starts a new session, without generating a response from the ai, then the session with only 1 input is deleted when the user switches to whatever other choice. Thus, there will always be 2 interactions in all present history slots, and it will generate the label from each of those slots with 2 interactions.    
5. Need THINK option to be visible when using reasoning model with any modes. THINK option should obviously not be visible if not a reasoning model. `Enable THINK`, enabled by default, but I think there is something special to put into the prompt or the argument when its a reasoning model and you do not want it to use the THINK phase.  
6. Attach files is uploading to the vectorstore, then having some error. I also notice that it immediately uploads the files, instead of waiting for the user to then click send, like traditional chatbots do, however, Possibly a better way to do vectorstores, would be to not have file slots as items, but instead a text display like...
```
Vectorized Data:
ExampleFilenameOne.tct
ExampleFilenameTwo.tct
ExampleFilenameThree.tct
```
...this make one wonder if a specified number of file slots should still be enforced at all, even as the total number of files allowed for the session to be entered into the database. When the user adds files, they are uploaded to the vectorstore, and for each file uploaded, this then becomes noted in the chatlog in a recognizable format, as the document/file limited to 38 characters with a `..` at the end unless the full filename fits in 38 characters. The listed items remaining listed in a display for the duration that session is open, because the database is session specific, so as to be able to display what is in the vector database relevant to the session. The idea is, by printing it in the session, then the program scanning the given session log through for files added, then compiling the information, and producing the list from it, then adding any additional ones to that list, instead of re-scanning the complete log repeatingly during a session. I would have to think about it. For notation in the Session log of attached files, it could be like...
```
File Vectorized: ExampleFilenameOne.txt
File Vectorized: ExampleFilenameTwo.txt
```
...on the other hand, if the processes of attaching files for purposes of input in AI chatbots requires the files be added to each interaction of relevance, and the data is not pulled from the vectorstore, then obviously we would want to keep it as file slots. I am unsure as to how it works, but if it is just about dumping data into a vector store while not requiring the original files after, then there are simpler more srtreamline methods of managine it on the UI, is what I am trying to say. Possibly even at the start of the session history we could maintain a list like...
```
Vectorized Data:
    ExampleFilenameOne.txt, ExampleFilenameTwo.txt, ExampleFilenameThree.txt
``` 
...that is read for populating the Vectorized Data display, instead of having to search through the document. The idea of populating the list somehow from the session history log, is to reduce required variables
## Far Development.
With regards to the next generation of the program will be ...
- Add `PyWin32` and `Whisper`, details are here.`https://www.perplexity.ai/search/1-how-does-mantella-mod-for-sk-Q32RILakTQ.lvQ3NHLJb5A`.
- There are a lot of globals, ensure all globals are safe_three_word labels, so as, to not so much have a chance to interfere with other programs using the same globals.
- attatched files should be able to be written to.
- 6 modes and dual model like this...
```
| **Mode**             | **Models**                     | **Features**                                      | **Implementation Notes**                                                                 |
|-----------------------|--------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------|
| Chatbot              | Single chat model (e.g., DeepSeek-V3) | TOT reasoning, web research                     | Use LangChain for TOT (e.g., recursive prompt splitting), SerpAPI for web queries       |
| Advanced Chatbot     | Fast + Quality chat models    | TOT on Fast, AUTO/FAST/QUALITY switch, web on Fast | Quantize Fast model (e.g., 4-bit), add Gradio slider for mode, cache Quality outputs   |
| Coder                | Single code model (e.g., DeepSeek-Coder) | Syntax, formatting, completion                | Leverage Pygments for highlighting, integrate VSCode-like keybinds                     |
| Agentic Coder        | Code + text models            | Intent → code → review loop                    | Chain via LangChain agents, store intermediate states in .\data\                       |
| RPG Text             | Single RPG model              | Narrative, character tracking, uncensored       | Parse entities with regex, save JSON states in .\data\history\                         |
| RPG Text + Images    | RPG + image model (Flux.1-schnell) | Text RPG + scene images                      | Trigger Flux via llama.cpp image fork, display in Gradio gallery (256x256 max)         |
```
- Ideas for introduce a, `Chat-Persistent` and `Rpg-Persistent`, mode, where under that mode then it is not specific to the session, and by creating/loading a session, then it would look this up. This could be like lighter version of RPG with 1 AI character, and be like, `Counsellor` or `Girlfriend`, then someone could persistently chat, they cant because its on rotation, to there is a point to it.
- Introduction of `Text-Gradio-Gguf.sh` file and modifications of scripts, to enable, Linux AND Windows, support. 
- Verbose Clear Concise Printed Notifications for all stages of model interaction/json handling: `Sending Prompt to Code Model...`, `Generating Code for Chat Model...`, `Response Received from Code Model...`.
- Color themes. Deepseek R1 AND Grok3Beta, found this extremely difficult, Deepseek was better. It needs to be re-attempted later, possibly with GPT4o1. Make it all shades of grey, then allow people to choose the primary button color in configuration, or something simple like that.
- two new modes—Chat-Notate and Chat-Notate-Uncensored—to your Windows 10-based chatbot. These modes allow the AI to process uploaded PDFs into summarized notations, stored in .\data\notation, with two versions: a detailed summary (up to the model's context limit, minus a safety margin) and a concise summary (256 tokens). During chats, the AI selects the most relevant detailed notation based on user input to deliver informed responses. A Notation Library menu in the UI enables users to manage notations by indexing new PDFs or deleting existing ones. For reasoning models in these modes, the THINK phase is disabled (like in TOT mode), ensuring practical, notation-driven conversations. This feature is slated for a later development phase.

## Credits
- [Grok3Beta](https://x.com/i/grok) - For much of the complete updated functions that I implemented.
- [Deepseek R1/3](https://www.deepseek.com/) - For re-attempting the things Grok3Beta was having difficulty with.
- [Claude_Sonnet](https://claude.ai) - For a smaller amount of the work on this project, useually when there was issue with the other 2 above.
- [Perplexity](https://www.perplexity.ai) - For research to look at more sites than normal, also notice its agentic now.

## License
**Wiseman-Timelord's Glorified License** is detailed in the file `.\Licence.txt`, that will later be added.

