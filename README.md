# Assistant with Voice Interaction 🤖⚽

This project combines **Generative AI** with football knowledge, leveraging **Retrieval-Augmented Generation (RAG)**, **Automatic Speech Recognition (ASR)**, and **Text-to-Speech (TTS)** technologies. The assistant allows users to ask questions about football rules and receive answers both in text and speech formats, based on the official rule book by The International Football Association Board (IFAB).

What's cool about this assistant is that you can **ask questions with your voice** and **hear** the AI's response, making it feel more interactive and engaging.


## Demo

I've deployed a demo of the project on a  **Hugging Face Spaces**, so feel free to check it out [here](https://huggingface.co/spaces/guirnd/rag-voice-assistant).

## Key Features

- **LLM-Powered ChatPDF**: Uses a **Large Language Model (LLM)** to answer your football-related queries based on the official rule book.
- **RAG (Retrieval-Augmented Generation)**: Retrieves the most relevant sections of the rule book to provide accurate, context-based responses.
- **Automatic Speech Recognition (ASR)**: You can ask questions by speaking, and the assistant will transcribe your speech using.
- **Text-to-Speech (TTS)**: The assistant converts its text responses back into speech, so you can hear the answers.
- **No Need for APIs**: Everything runs locally with open-source models, so no external APIs are required.
- **Easily Adaptable to Other Domains**: You can swap out the football rule book with any other PDF (healthcare, law, education) and have the assistant answer questions based on that new content.

## Technologies Used

- **LLM**: Powered by **mistral-7b-instruct** for efficient, CPU-based natural language generation.
- **ASR**: Uses **Whisper** for speech-to-text transcription.
- **TTS**: Utilizes **SpeechT5** for natural-sounding speech synthesis.
- **RAG**: Retrieval-Augmented Generation using **FAISS** for document retrieval and **all-MiniLM-L6-v2** for embeddings.
- **LangChain**: Orchestrates the LLM and retrieval mechanisms to provide accurate responses.
- **Gradio**: Provides a simple, intuitive interface to interact with the AI.

  ## Installation

### Clone the Repository

```bash
git clone https://github.com/guirn/football-ai-assistant.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

This will install everything you need, including **LangChain**, **FAISS**, **Gradio**, **transformers**, and **datasets**.

### Download the Models

The project uses open-source models available on Hugging Face. You can download them as needed:

- [mistral-7b-instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [Whisper](https://huggingface.co/openai/whisper-small)
- [SpeechT5](https://huggingface.co/microsoft/speecht5_tts)
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Football rules PDF

The Laws of the Game PDF can be downloaded from [IFAB's](https://www.theifab.com/laws-of-the-game-documents/?language=all&year=2024%2F25) website.

Disclaimer: This application was developed solely for educational purposes to demonstrate AI capabilities and should not be used as a source of information or for any other purpose.

### Adaptation to Other Domains

While this project focuses on football, it's super easy to adapt it for other topics. All you need to do is replace the football rule book PDF with a new one, such as medical guidelines, legal documents, or any other subject-and the assistant will start answering questions based on that content.

### Contribution

If you have ideas, improvements, or want to help out, I'd love to see your contributions! Open an issue or submit a pull request.

### License

This project is open-source under the MIT License.
