from pathlib import Path
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file_path: str) -> str:
    """Unstructured handles PDF, DOCX, images, Excel — one unified call."""
    elements = partition(filename=file_path)
    return "\n".join([str(el) for el in elements])

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)