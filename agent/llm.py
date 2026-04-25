"""
LLM factory — single place to change models.
Swap to a vLLM-backed open-source model by changing this file only.
"""

import asyncio
from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings

# Global semaphore to cap concurrent LLM calls
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.llm_semaphore)
    return _semaphore


@lru_cache(maxsize=4)
def get_llm(tier: Literal["heavy", "light"] = "heavy", temperature: float = 0.0) -> ChatOpenAI:
    model = settings.heavy_model if tier == "heavy" else settings.light_model
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
