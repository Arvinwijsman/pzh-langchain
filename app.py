import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from utils.settings import (
    settings,
    SystemStatus,
    test_openai_connection,
    test_redis_connection,
)
from utils.file_processing import extract_document_list, scan_documents_folder
from utils.redis import build_redis_vectorstore

# Init and health check
load_dotenv()
st.set_page_config(page_title="STOP/TPOD documentatie AI CHAT")
redis_status = test_redis_connection(settings.redis_host, settings.redis_port)
openai_status = test_openai_connection(settings.openai_api_key)
system_status = SystemStatus(
    redis_connection=redis_status,
    openai_connection=openai_status,
    files_in_folder=scan_documents_folder(settings.documents_path),
)
st.session_state.sys_status = system_status


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def process_documents():
    full_content = extract_document_list(system_status.files_in_folder)
    redis_vs = build_redis_vectorstore(full_content, redis_index="pzh")


#### SIDEBAR CONTROL PANEL
with st.sidebar:
    st.header("System Status")
    redis_conn_state = st.text("REDIS: ❌")
    if system_status.redis_connection is True:
        redis_conn_state.text("REDIS: ✅")

    openai_conn_state = st.text("OPENAI: ❌")
    if system_status.openai_connection is True:
        openai_conn_state.text("OPENAI: ✅")

    st.divider()

    st.header("Documents")
    st.subheader(f"Found files in {settings.documents_path}:")
    # st.text(("\n".join(system_status.files_in_folder)))
    for file in system_status.files_in_folder:
        filename = file.replace(settings.documents_path, "")
        st.checkbox(filename, value=True, key=file)

    st.subheader("Learned/Processed Documents")
    st.button("Process Documents", on_click=process_documents())
    # for doc in system_status.loaded_documents:
    #     use = st.checkbox(doc, value=True)

    st.divider()
    if st.button("Clear Cache"):
        st.cache_data.clear()

    st.divider()
    if st.button("Settings Dump"):
        st.code(settings.json())


# Main Page
st.header("AI Chat STOP/TPOD Documentatie :books:")
st.subheader("PZH Omgevingsbeleid")
