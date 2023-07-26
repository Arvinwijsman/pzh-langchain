from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Redis
from langchain.embeddings import OpenAIEmbeddings

from settings import settings


# Vectorestore / Redis
def build_redis_vectorstore(content: str, redis_index: str = "pzh"):
    chunked = CharacterTextSplitter(
        separator="\n",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    ).split_text(content)

    embeddings = OpenAIEmbeddings()

    rds = Redis.from_texts(
        texts=chunked,
        embedding=embeddings,
        redis_url=settings.redis_dsn,
        index_name=redis_index,
    )
    return rds
