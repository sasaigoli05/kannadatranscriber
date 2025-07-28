import streamlit as st
import torch
from transformers import pipeline

st.title("Kannada Audio Transcription")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            device = "cpu"
            transcribe = pipeline(
                task="automatic-speech-recognition",
                model="vasista22/whisper-kannada-tiny",
                chunk_length_s=30,
                device=0 if device == "cuda" else -1  # -1 tells HF to use CPU
            )
            transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="kn", task="transcribe")
            
            # Save to temp file
            with open("temp_audio.mp3", "wb") as f:
                f.write(uploaded_file.read())

            result = transcribe("temp_audio.mp3")["text"]

            # Save result
            with open("transcription_output.txt", "w", encoding="utf-8") as f:
                f.write(result)

            st.success("Transcription complete!")
            st.text_area("Transcript:", result, height=200)
            st.download_button("Download Transcription", result, file_name="transcription_output.txt")
