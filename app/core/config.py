from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    TESSERACT_CMD: str = r"C:\Users\JatinSharma\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    VISION_MODEL: str = "llava"

    class Config:
        env_file = ".env"

settings = Settings()