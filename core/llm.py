from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from core.config import settings


def create_llm():
    return ChatMistralAI(
        model_name=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        max_retries=2,
    )


def create_embeddings():
    return MistralAIEmbeddings(model=settings.EMBED_MODEL)
