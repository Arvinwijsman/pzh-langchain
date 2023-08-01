from typing import List
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores.docarray import DocArrayInMemorySearch
import redis
from langchain.vectorstores.redis import Redis, _check_index_exists
from langchain.embeddings import OpenAIEmbeddings

from .settings import settings
from utils.file_processing import preprocess_documents


def _init_redis_connection(db=0):
    return redis.Redis(host=settings.redis_host, port=settings.redis_port, db=db)


def reset_redis_db():
    rds = _init_redis_connection()
    return rds.flushall()


# Refactor from pzh
def setup_redis_vectorstore(redis_index="link"):
    """
    Embed documents and builds vectorstore to redis.
    if the index already exists in the redis db,
    skip document embedding and just return the vectorestore
    instance.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=settings.openai_api_key,
        chunk_size=settings.chunk_size,
    )

    rds = _init_redis_connection()

    if _check_index_exists(rds, redis_index):
        print(f"Found existing index: {redis_index} in redis.")
        return Redis(
            redis_url=settings.redis_dsn,
            index_name=redis_index,
            embedding_function=embeddings.embed_query,
        )

    print(f"No existing index: {redis_index} found. Embedding new documents.")

    # PDFs -> Chunk list
    document_chunks = preprocess_documents()

    # Chunk list -> Embed -> Redis vectorstore
    return Redis.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        redis_url=settings.redis_dsn,
        index_name=redis_index,
    )


def setup_memory_vectorstore(_chunks: List[Document]):
    """
    Alternative in memory vstore, easy to deploy, lower performance.
    # TODO: Compare with redis / FAISS / Embeddings
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(_chunks, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )
    return retriever


def setup_faiss_vectorstore(_chunks: List[Document]):
    # TODO: try and compare performance
    raise NotImplementedError
