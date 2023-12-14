import streamlit as st
import requests
from PyPDF2 import PdfReader
import io


API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def extract_text_from_pdf(uploaded_file):
    pdf_data = io.BytesIO(uploaded_file.read())
    pdf_reader = PdfReader(pdf_data)

    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

    return text


def split_text_into_chunks(text, lines_per_chunk=20):
    lines = text.split('\n')
    chunks = ['\n'.join(lines[i:i+lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
    return chunks


def main():
    st.title("PDF Summarization with BART")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("PDF File Uploaded.")
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(pdf_text)
        summary_text = ""
        for chunk in chunks:
            output = query({"inputs": chunk})
            if isinstance(output, list) and len(output) > 0:
                summary_text += output[0]['summary_text'] + "\n\n"
        st.header("Summary:")
        if summary_text:
            with st.container():
                st.markdown(summary_text)
        else:
            st.text("Error: Unable to fetch the summary.")


if __name__ == "__main__":
    main()
