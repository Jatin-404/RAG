import json
import pandas as pd
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
import base64
import requests
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import pytesseract
from pytesseract.pytesseract import TesseractNotFoundError
from app.core.config import settings
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# ─── File type routing ────────────────────────────────────────────────

PLAIN_TEXT_TYPES  = {".txt", ".md"}
TABULAR_TYPES     = {".csv", ".xlsx", ".xls", ".ods"}
JSON_TYPES        = {".json"}
IMAGE_TYPES       = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
OFFICE_TYPES      = {".docx", ".odt", ".pptx", ".ppt", ".doc"}
PDF_TYPE          = {".pdf"}

# ─── Handlers ─────────────────────────────────────────────────────────

def _handle_plain_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _handle_json(file_path: str) -> str:
    """Extracts text content from JSON. Handles both flat and nested."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # known text fields in order of preference
    for key in ["text", "content", "body", "description"]:
        if key in data and isinstance(data[key], str):
            return data[key]

    # fallback — flatten everything to string
    return json.dumps(data, indent=2)

def _handle_tabular(file_path: str) -> str:
    """
    Full Excel/CSV pipeline:
    - Multi-sheet support
    - Schema detection
    - Row-level natural language conversion
    - Table summary generation
    Returns combined text of all sheets.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == ".csv":
        sheets = {"Sheet1": pd.read_csv(file_path)}
    else:
        xl = pd.ExcelFile(file_path)
        sheets = {
            sheet: xl.parse(sheet) 
            for sheet in xl.sheet_names
        }
    
    all_text = []
    
    for sheet_name, df in sheets.items():
        df = df.fillna("").astype(str)
        columns = list(df.columns)
        
        if not columns or df.empty:
            continue
        
        # 1. Table summary
        summary = _generate_table_summary(sheet_name, columns, df)
        all_text.append(f"[TABLE_SUMMARY sheet={sheet_name}]\n{summary}")
        
        # 2. Row-level natural language
        for _, row in df.iterrows():
            row_text = _row_to_natural_language(sheet_name, row, columns)
            if row_text.strip():
                all_text.append(f"[ROW sheet={sheet_name}]\n{row_text}")
    
    return "\n\n".join(all_text)


def _generate_table_summary(sheet_name: str, columns: list, df: pd.DataFrame) -> str:
    """
    Uses Ollama to generate a semantic summary of the table.
    This helps answer broad queries like 'what data is available?'
    """
    sample_rows = df.head(3).to_string(index=False)
    
    prompt = f"""You are analyzing a spreadsheet sheet called "{sheet_name}".
Columns: {', '.join(columns)}
Sample data:
{sample_rows}

Write a single paragraph describing what this table contains, what kind of records it stores, and what questions it can answer. Be concise."""

    try:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception:
        # Fallback if LLM fails
        return f"This table contains {len(df)} records with columns: {', '.join(columns)}."


def _row_to_natural_language(sheet_name: str, row: pd.Series, columns: list) -> str:
    """
    Converts a row to natural language.
    'Name: John | Age: 30 | Salary: 50000 | Department: HR'
    """
    parts = []
    for col in columns:
        val = str(row[col]).strip()
        if val and val != "nan":
            parts.append(f"{col}: {val}")
    
    return " | ".join(parts)

def _handle_office_chunked(file_path: str) -> list[str]:
    elements = partition(filename=file_path)
    doc_text = "\n".join([str(el) for el in elements]).strip()

    image_text = _extract_embedded_images(file_path)

    if not image_text:
        print("⚠️ No image text extracted")

    chunks = []

    if doc_text:
        chunks.extend([f"[TEXT]\n{c}" for c in chunk_text(doc_text)])

    if image_text:
        records = [
            block.strip()
            for block in image_text.split("\n\n")
            if block.strip()
        ]
        chunks.extend([f"[ROW]\n{r}" for r in records])

    return chunks if chunks else [""]


def extract_chunks(file_path: str) -> list[str]:
    ext = Path(file_path).suffix.lower()

    if ext in TABULAR_TYPES:
        return _handle_tabular_chunked(file_path)
    elif ext in OFFICE_TYPES:
        return _handle_office_chunked(file_path)
    else:
        text = extract_text(file_path)
        return chunk_text(text)


def _handle_tabular_chunked(file_path: str) -> list[str]:
    """Returns one chunk per row + one summary per sheet."""
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        sheets = {"Sheet1": pd.read_csv(file_path)}
    else:
        xl = pd.ExcelFile(file_path)
        sheets = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

    chunks = []

    for sheet_name, df in sheets.items():
        df = df.fillna("").astype(str)
        columns = list(df.columns)

        if not columns or df.empty:
            continue

        # One summary chunk per sheet
        summary = _generate_table_summary(sheet_name, columns, df)
        chunks.append(f"[TABLE_SUMMARY sheet={sheet_name}]\n{summary}")

        # One chunk per row
        for _, row in df.iterrows():
            row_text = _row_to_natural_language(sheet_name, row, columns)
            if row_text.strip():
                chunks.append(f"[ROW sheet={sheet_name}]\n{row_text}")

    return chunks



def _handle_image(file_path: str) -> str:
    """Uses LLaVA vision model for all standalone images."""
    from PIL import Image
    import io

    img = Image.open(file_path)
    img.load()
    img_copy = img.copy()
    img.close()
    return _vision_extract_image(img_copy)
    

def _handle_pdf(file_path: str) -> str:
    """
    Tries digital extraction first.
    Falls back to OCR if no text found (scanned PDF).
    """
    # Import lazily so the API can start without PDF extras installed.
    # `unstructured.partition.pdf` may require optional deps like `unstructured_inference`.
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except Exception:
        # Fallback to the generic partitioner (works for many PDFs if deps are present)
        # and otherwise provide a clear actionable error only when a PDF is processed.
        try:
            elements = partition(filename=file_path)
            return "\n".join([str(el) for el in elements]).strip()
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PDF parsing requires optional Unstructured dependencies. "
                'Install with: `uv add "unstructured[pdf]"` (or `pip install "unstructured[pdf]"`).'
            ) from e

    # Try fast digital extraction first
    elements = partition_pdf(filename=file_path, strategy="fast")
    text = "\n".join([str(el) for el in elements]).strip()

    # If no text extracted — it's a scanned PDF, use OCR
    if not text:
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
        )
        text = "\n".join([str(el) for el in elements])

    return text

import pandas as pd

def _vision_extract_image(img) -> str:
    """
    Uses LLaVA vision model to extract text from images.
    Handles tables, scanned docs, mixed content far better than OCR.
    """
    # Convert PIL image to base64
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = """Extract structured data from this document image.
                STRICT RULES:
                - Each row MUST be separated by exactly ONE blank line
                - Each row MUST represent ONE record only
                - Format: ColumnName: Value | ColumnName: Value
                - Do NOT merge multiple rows
                - Do NOT skip any values
                - If unsure, still output best guess

                Return ONLY the extracted data."""

    response = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.VISION_MODEL,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def _extract_embedded_images(file_path: str) -> str:
    extracted_texts = []
    image_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    try:
        with zipfile.ZipFile(file_path, "r") as z:
            image_files = [
                f for f in z.namelist()
                if Path(f).suffix.lower() in image_extensions
            ]

            for image_file in image_files:
                with z.open(image_file) as img_data:
                    image_data = img_data.read()

                # Write to fixed path, fully closed before PIL touches it
                tmp_path = os.path.join(
                    tempfile.gettempdir(), f"rag_ocr_{os.getpid()}.png"
                )
                with open(tmp_path, "wb") as f:
                    f.write(image_data)

                try:
                    img = Image.open(tmp_path)
                    img.load()
                    img_copy = img.copy()
                    img.close()
                    ocr_text = _vision_extract_image(img_copy)
                    if ocr_text:
                        extracted_texts.append(ocr_text)
                except Exception as e:
                    print("OCR ERROR:", type(e).__name__, str(e))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    except zipfile.BadZipFile:
        pass

    print("ZIP CONTENTS:", z.namelist())
    print("IMAGE FILES FOUND:", image_files)

    return "\n".join(extracted_texts)

def _handle_office(file_path: str) -> str:
    """
    Handles DOCX, ODT, PPTX.
    Also extracts and OCRs any embedded images inside the document.
    """
    # First get the normal text
    elements = partition(filename=file_path)
    text = "\n".join([str(el) for el in elements]).strip()

    # Then extract embedded images and OCR them
    embedded_text = _extract_embedded_images(file_path)

    return f"{text}\n{embedded_text}".strip()

# ─── Main dispatcher ──────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """
    Single entry point for all file types.
    Routes to the right handler based on extension.
    """
    ext = Path(file_path).suffix.lower()

    if ext in PLAIN_TEXT_TYPES:
        return _handle_plain_text(file_path)
    elif ext in JSON_TYPES:
        return _handle_json(file_path)
    elif ext in TABULAR_TYPES:
        return _handle_tabular(file_path)
    elif ext in IMAGE_TYPES:
        return _handle_image(file_path)
    elif ext in PDF_TYPE:
        return _handle_pdf(file_path)
    elif ext in OFFICE_TYPES:
        return _handle_office(file_path)
    else:
        # Unknown type — let Unstructured try its best
        try:
            elements = partition(filename=file_path)
            return "\n".join([str(el) for el in elements])
        except Exception as e:
            raise ValueError(f"Unsupported file type: {ext}. Error: {str(e)}")

# ─── Metadata extractor for JSON ──────────────────────────────────────

def extract_json_metadata(file_path: str) -> dict:
    """
    Pulls structured metadata from JSON files.
    Handles your Indian court judgment format + generic JSON.
    """
    if Path(file_path).suffix.lower() not in JSON_TYPES:
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Indian court judgment format
    if "judgment_id" in data:
        meta = data.get("metadata", {})
        classification = data.get("classification", {})
        return {
            "judgment_id": data.get("judgment_id", ""),
            "court": meta.get("court"),
            "court_level": meta.get("court_level"),
            "decision_date": meta.get("decision_date"),
            "bench": meta.get("bench"),
            "jurisdiction": meta.get("jurisdiction"),
            "detected_domain": classification.get("domain", "legal")
        }

    # Generic JSON — return top level string fields as metadata
    return {
        k: v for k, v in data.items()
        if isinstance(v, str) and k != "text" and k != "content"
    }

# ─── Chunker ──────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)