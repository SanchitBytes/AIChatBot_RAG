import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DB_PATH = os.getenv("DB_PATH", "chatbot.db")
    MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral-embed")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 200))


settings = Settings()
