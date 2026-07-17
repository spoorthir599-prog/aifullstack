from transformers import pipeline
import streamlit as st
@st.cache_resource
def load_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

qa = load_model()
st.title("Python Chatbot")
question = st.text_input("Ask a question")
python_notes = """
Python is a high level language.
Python can be understood by everyone.
Python contains variables and functions.
Python is used for web development, data science, and automation.
"""
if st.button("Get Answer"):
    if question:
        result = qa(
            context=python_notes,
            question=question,
            max_answer_len=10
        )

        st.write("Answer:")
        st.success(result["answer"])

    else:
        st.warning("Please enter a question")