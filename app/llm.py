"""Connection to the language model (OpenAI or Azure OpenAI).

Everything that talks to the model goes through this file, so switching
between OpenAI and Azure OpenAI only needs changes in your .env file.
"""
import os

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

_client = None


def get_client():
    """Return one shared client (created once, reused for every call)."""
    global _client
    if _client is None:
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            # Azure OpenAI: CHAT_MODEL / EMBEDDING_MODEL are your *deployment* names.
            _client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into embeddings (lists of numbers that represent meaning)."""
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
