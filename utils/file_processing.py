import os
from typing import List, Optional
import streamlit as st
from PyPDF2 import PdfReader


# UTILS
def extract_pdf_text(file_path: str):
    text = ""
    with open(file_path, "rb") as file:
        pdf = PdfReader(file)
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()

    return text


def scan_documents_folder(folder_path: str) -> List[str]:
    file_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_paths.append(os.path.join(folder_path, filename))
    return file_paths


@st.cache_data
def extract_document_list(paths: List[str]) -> str:
    combined_content = ""
    for path in paths:
        try:
            combined_content += extract_pdf_text(path)
        except Exception as e:
            st.warning(f"Failed to read file: {path}", icon="⚠️")
            print(e)

    return combined_content


def chuck_splitter(text):
    CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)
