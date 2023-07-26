from typing import List, Optional
from pydantic import BaseSettings, BaseModel
import redis
from redis.exceptions import ConnectionError
import openai
import streamlit as st


class Settings(BaseSettings):
    openai_api_key: str
    redis_host: str = "localhost"
    redis_port: int = 6379
    documents_path: str = "./documents/"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def redis_dsn(self):
        return f"redis://{self.redis_host}:{self.redis_port}"

    class Config:
        env_file = ".env"


class SystemStatus(BaseModel):
    redis_connection: bool = False
    openai_connection: bool = False
    files_in_folder: List[str] = []
    loaded_documents: List[str] = []


# Instantiate
settings = Settings()


def test_redis_connection(host, port, db=0):
    try:
        r = redis.Redis(host=host, port=port, db=db)
        r.ping()
        return True
    except ConnectionError:
        return False


@st.cache_data
def test_openai_connection(api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003", prompt="test", max_tokens=5
        )
        return True
    except Exception as e:
        print(f"Connection failed with error: {e}")
        return False
