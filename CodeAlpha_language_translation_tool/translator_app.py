import streamlit as st
from deep_translator import GoogleTranslator
import pyperclip

st.title("🌍 Language Translation Tool")
st.write("Translate text between different languages using AI")

languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Japanese": "ja",
    "Chinese": "zh-CN"
}

input_text = st.text_area("Enter text to translate")

source_lang = st.selectbox("Select Source Language", languages.keys())
target_lang = st.selectbox("Select Target Language", languages.keys())

if st.button("Translate"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        translated_text = GoogleTranslator(
            source=languages[source_lang],
            target=languages[target_lang]
        ).translate(input_text)

        st.subheader("Translated Text:")
        st.success(translated_text)

        if st.button("Copy Translated Text"):
            pyperclip.copy(translated_text)
            st.info("Text copied to clipboard!")

# to run file 
# python -m streamlit run translator_app.py
