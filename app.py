import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain

from utils.settings import settings
from utils.file_processing import scan_documents_folder
from utils.ping import ping_redis, ping_openai
from utils.vectorstores import (
    reset_redis_db,
    setup_redis_vectorstore,
    _init_redis_connection,
)

st.set_page_config(page_title=settings.home_title, page_icon="🤖")

# Intro content
st.header("LAZY-TPOD :books:")
st.write(
    """
    Deze app bevat een ChatGPT-like interface met de GPT-3.5 API en kan documentatie van STOP/TPOD 
    verwerken in LLM prompts als context. Bij het geven van prompts zal de chat automatisch proberen antwoorden
    over de documenten te geven indien mogelijk.

    Notes:

    - Beschikbare / ingeladen documenten staan in de sidebar.
    - STOP documentatie moet nog toegevoegd.
    - Dit is grove opzet, nog niet optimized en resultaten zijn niet altijd pluis.
    - Conversational memory werkt, dus doorvragen op vorige antwoorden is mogelijk.
    """
)
st.divider()


@st.cache_resource(ttl="1h")
def initialize_chain():
    vstore = setup_redis_vectorstore()

    # TODO: compare? "similarity_limit", "similarity"
    retriever = vstore.as_retriever(search_type="similarity")

    # Buffer to store and return the chat history as prompt context
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Use streaming arg for ChatGPT model with custom callback handlers
    llm = ChatOpenAI(
        model_name=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=settings.openai_temperature,
        streaming=True,
    )

    # Using standard qa chain with document stuffing
    # TODO: Try custom QA prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        verbose=True,  # print the final prompts to console
        chain_type="stuff",
    )

    return qa_chain


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Bekijk document context")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


def nuke_cache():
    st.cache_resource.clear()
    st.session_state.clear()
    reset_redis_db()


def initialize_session():
    st.session_state.redis_ping = ping_redis()
    st.session_state.openai_ping = ping_openai()
    st.session_state.local_documents = scan_documents_folder(settings.documents_path)
    st.toast("New session initialized")


# MAIN
state_vars = ["redis_ping", "openai_ping", "local_documents"]
if any(var not in st.session_state for var in state_vars):
    initialize_session()

qa_chain = initialize_chain()

# Build sidebar
with st.sidebar:
    st.title("DEV SIDEBAR")
    st.header("Connection Status")
    st.text("REDIS: ❌" if not st.session_state.redis_ping else "REDIS: ✅")
    st.text("OPENAI: ❌" if not st.session_state.openai_ping else "OPENAI: ✅")

    st.divider()

    st.header("Document Status")
    st.text(f"Found files in {settings.documents_path}:")
    for file in st.session_state.local_documents:
        filename = file.replace(settings.documents_path, "")
        st.write(filename)
        # st.checkbox(filename, value=True, key=file)

    st.divider()

    if st.button("Log session state"):
        print(st.session_state)

    if st.button("Reset Session + Redis"):
        nuke_cache()
        initialize_session()
        st.toast("done resetting, building new vectorstore...")
        qa_chain = initialize_chain()
        st.toast("Ready")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hoi! Vraag me alles over STOP-TPOD! maar niet te veel want arvin's credit card is gekoppeld aan de GPT requests.",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(
    placeholder="Stel je vraag. Wees voor zo ver mogelijk specifiek."
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
