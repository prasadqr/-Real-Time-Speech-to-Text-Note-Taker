import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import tempfile
import base64
from openai import AzureOpenAI  # ✅ New import
from config import *
from openai import AzureOpenAI

# Set up OpenAI client (NEW SDK FORMAT)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Streamlit UI
st.set_page_config(page_title="🎤 Audio to Bullet Points", layout="centered")
st.title("🎤 Audio File → 📌 Bullet Points using Phi-4")

st.markdown("Upload a WAV audio file (mono, 16kHz). We’ll transcribe and summarize it using Azure services.")

uploaded_file = st.file_uploader("📁 Upload Audio File", type=["wav"])

if uploaded_file:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.info("🔄 Transcribing with Azure Speech...")

    # Transcribe audio using Azure Speech SDK
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text
        st.subheader("📝 Transcript")
        st.write(transcript)

        st.subheader("📌 Summarizing with Phi-4")

        prompt = f"Summarize the following into concise bullet points:\n\n{transcript}"

        # 🔄 Use the new SDK method
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that make some notes from audio transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        summary = response.choices[0].message.content
        st.success("🔍 Bullet Points Summary:")
        st.markdown(summary)

    else:
        st.error("❌ Could not recognize speech. Try another file.")
