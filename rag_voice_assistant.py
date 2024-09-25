import os
import re
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQA
from langchain.llms import LlamaCpp
import gradio as gr

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_and_split_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        return docs

class FAISSManager:
    def __init__(self):
        self.vectorstore_cache = {}

    def build_faiss_index(self, docs):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    def save_faiss_index(self, vectorstore, file_path):
        vectorstore.save_local(file_path)
        print(f"Vectorstore saved to {file_path}")

    def load_faiss_index(self, file_path):
        if not os.path.exists(f"{file_path}/index.faiss") or not os.path.exists(f"{file_path}/index.pkl"):
            raise FileNotFoundError(f"Could not find FAISS index or metadata files in {file_path}")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Vectorstore loaded from {file_path}")
        return vectorstore

    def build_faiss_index_with_cache_and_file(self, pdf_processor, vectorstore_path):
        if os.path.exists(vectorstore_path):
            print(f"Loading vectorstore from file {vectorstore_path}")
            return self.load_faiss_index(vectorstore_path)

        print(f"Building new vectorstore for {pdf_processor.pdf_path}")
        docs = pdf_processor.load_and_split_pdf()
        vectorstore = self.build_faiss_index(docs)
        self.save_faiss_index(vectorstore, vectorstore_path)
        return vectorstore

class LLMChainFactory:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def create_llm_chain(self, llm, max_tokens=80):
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["documents", "question"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        llm_chain.llm.max_tokens = max_tokens
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="documents"
        )
        return combine_documents_chain

class LLMManager:
    def __init__(self, model_path):
        self.llm = LlamaCpp(model_path=model_path)
        self.llm.max_tokens = 80

    def create_rag_chain(self, llm_chain_factory, vectorstore):
        retriever = vectorstore.as_retriever()
        combine_documents_chain = llm_chain_factory.create_llm_chain(self.llm)
        qa_chain = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)
        return qa_chain

    def main_rag_pipeline(self, pdf_processor, query, vectorstore_manager, vectorstore_file):
        vectorstore = vectorstore_manager.build_faiss_index_with_cache_and_file(pdf_processor, vectorstore_file)
        llm_chain_factory = LLMChainFactory(prompt_template="""You are a helpful AI. Based on the context below, answer the question politely.
        Context: {documents}
        Question: {question}
        Answer:""")
        rag_chain = self.create_rag_chain(llm_chain_factory, vectorstore)
        result = rag_chain.run(query)
        return result

class WhisperManager:
    def __init__(self):
        self.model_id = "openai/whisper-small"
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.whisper_processor = WhisperProcessor.from_pretrained(self.model_id)
        self.forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(language="english", task="transcribe")

    def transcribe_speech(self, filepath):
        if not os.path.isfile(filepath):
            raise ValueError(f"Invalid file path: {filepath}")
        waveform, sample_rate = torchaudio.load(filepath)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        input_features = self.whisper_processor(waveform.squeeze(), sampling_rate=target_sample_rate, return_tensors="pt").input_features
        generated_ids = self.whisper_model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
        transcribed_text = self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        cleaned_text = re.sub(r"<[^>]*>", "", transcribed_text).strip()
        return cleaned_text

class SpeechT5Manager:
    def __init__(self):
        self.SpeechT5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.SpeechT5_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_embedding_model = AutoModel.from_pretrained("microsoft/speecht5_vc")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.pretrained_speaker_embeddings = torch.tensor(embeddings_dataset[7000]["xvector"]).unsqueeze(0)

    def text_to_speech(self, text, output_file="output_speechT5.wav"):
        inputs = self.SpeechT5_processor(text=[text], return_tensors="pt")
        speech = self.SpeechT5_model.generate_speech(inputs["input_ids"], self.pretrained_speaker_embeddings, vocoder=self.vocoder)
        sf.write(output_file, speech.numpy(), 16000)
        return output_file

# --- Gradio Interface ---
def asr_to_text(audio_file):
    transcribed_text = whisper_manager.transcribe_speech(audio_file)
    return transcribed_text

def process_with_llm_and_tts(transcribed_text):
    response_text = llm_manager.main_rag_pipeline(pdf_processor, transcribed_text, vectorstore_manager, vectorstore_file)
    audio_output = speech_manager.text_to_speech(response_text)
    return response_text, audio_output

# Instantiate Managers
pdf_processor = PDFProcessor('./files/LawsoftheGame2024_25.pdf')
vectorstore_manager = FAISSManager()
llm_manager = LLMManager(model_path="./files/mistral-7b-instruct-v0.2.Q2_K.gguf")
whisper_manager = WhisperManager()
speech_manager = SpeechT5Manager()
vectorstore_file = "./vectorstore_faiss"

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>RAG Powered Voice Assistant</h1>") #removed emojis
    gr.Markdown("<h1 style='text-align: center;'>Ask me anything about the rules of Football!</h1>")

    # Step 1: Audio input and ASR output
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Speak your question")
        asr_output = gr.Textbox(label="ASR Output (Edit if necessary)", interactive=True)

    # Button to process audio (ASR)
    asr_button = gr.Button("1 - Transform Voice to Text")

    # Step 2: LLM Response and TTS output
    with gr.Row():
        llm_response = gr.Textbox(label="LLM Response")
        tts_audio_output = gr.Audio(label="TTS Audio")

    # Button to process text with LLM
    llm_button = gr.Button("2 - Submit Question")

    # When ASR button is clicked, the audio is transcribed
    asr_button.click(fn=asr_to_text, inputs=audio_input, outputs=asr_output)

    # When LLM button is clicked, the text is processed with the LLM and converted to speech
    llm_button.click(fn=process_with_llm_and_tts, inputs=asr_output, outputs=[llm_response, tts_audio_output])

    # Disclaimer
    gr.Markdown(
        "<p style='text-align: center; color: gray;'>Disclaimer: This application was developed solely for educational purposes to demonstrate AI capabilities and should not be used as a source of information or for any other purpose.</p>"
    )

demo.launch(debug=True)