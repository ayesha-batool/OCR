import os
import tempfile
import time
from base64 import b64encode
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

# Folder containing images
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".docx"}
LEGACY_WORD_EXTENSIONS = {".doc"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}




# Gemini (used for image extraction)
FREE_MODELS = [
# 200
"gemini-flash-latest",
"gemini-2.5-flash",
"gemini-robotics-er-1.6-preview",
"gemma-3-27b-it",
"gemma-4-26b-a4b-it",
"gemma-4-31b-it",
"gemini-2.5-flash-lite",
"gemma-3-4b-it",
"gemma-3-12b-it",
"gemma-3-27b-it",
"gemma-4-26b-a4b-it",
"gemma-4-31b-it",
"gemini-3-flash-preview",

# 503
"gemini-flash-lite-latest",
"gemini-3.1-flash-lite-preview",

# 429
"gemini-2.0-flash-lite",
"gemini-2.5-pro-preview-tts",
"gemini-pro-latest",
"gemini-2.5-flash-image",
"gemini-2.5-pro",
"gemini-2.0-flash",
"gemini-2.0-flash-001",
"gemini-2.0-flash-lite-001",
"gemini-2.5-computer-use-preview-10-2025",
"gemini-3-pro-preview",
"gemini-3.1-pro-preview",
"gemini-3.1-pro-preview-customtools",
"gemini-3.1-flash-image-preview",
"nano-banana-pro-preview",
"gemini-3-pro-image-preview",
"lyria-3-clip-preview",
"lyria-3-pro-preview",


   
]
API_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 12)]
API_KEYS = [k for k in API_KEYS if k]
PROCESSOR_POST_ENDPOINT = os.getenv("PROCESSOR_POST_ENDPOINT", "").strip()
PROCESSOR_GET_ENDPOINT = os.getenv("PROCESSOR_GET_ENDPOINT", "").strip()


app = FastAPI(title="Universal Text Extractor", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SendExtractedPayload(BaseModel):
    endpoint: HttpUrl
    text: str


class ProcessExtractedPayload(BaseModel):
    text: str



def _extract_with_gemini(path: Path) -> str:
    print(f"\n📸 Image: {path.name}")

    if not API_KEYS:
        raise RuntimeError("Gemini API keys not configured. Set GEMINI_API_KEY_1..GEMINI_API_KEY_7.")
    if not FREE_MODELS:
        raise RuntimeError("No Gemini models configured.")

    image_b64 = b64encode(path.read_bytes()).decode("utf-8")

    mime = "image/jpeg"
    if path.suffix.lower() == ".png":
        mime = "image/png"

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "You are a strict OCR engine.\n"
                            "Extract text VERBATIM from the image.\n"
                            "Rules (must follow exactly):\n"
                            "- Do NOT translate, summarize, or paraphrase\n"
                            "- Do NOT fix spelling/grammar\n"
                            "- Keep Urdu/Arabic exactly as written\n"
                            "- Preserve original line breaks and punctuation\n"
                            "- Preserve repeated words and apparent mistakes\n"
                                  "dont write anything other than final output, add no workin like double checking or rewriting rules and i want exact text"
                            "i dont want checking. i only want text which is extracted \n" 
                            "dont rewrite rules or double check or add any explanation. just give me the text which is in image and nothing else\n"
                            "- Output ONLY extracted text, no explanation"
                        )
                    },
                    {"inline_data": {"mime_type": mime, "data": image_b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "topP": 0,
            "topK": 1,
            "responseMimeType": "text/plain",
            "maxOutputTokens": 8192
        },
    }

    last_error: Exception | None = None

    for api_index, api_key in enumerate(API_KEYS, start=1):
        for model in FREE_MODELS:
            print(f"🔑 API {api_index} | 🤖 {model}")
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": api_key},
                    json=body,
                    timeout=120,
                )
                print(f"🌐 Status: {r.status_code}")
                if r.status_code == 429:
                    print("🚫 Rate limit -> trying next key/model")
                    continue
                r.raise_for_status()
                data = r.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                if text:
                    print("\n================ OCR OUTPUT ================")
                    print(text)
                    print("===========================================\n")
                    return text
                print("⚠️ Empty response")
            except Exception as exc:
                print(f"❌ Exception: {exc}")
                last_error = exc
                continue

    if last_error:
        print("❌ Failed image")
        raise RuntimeError(f"Gemini extraction failed across all model/API combinations: {last_error}") from last_error
    print("❌ Failed image")
    raise RuntimeError("Gemini extraction failed across all model/API combinations.")


def process_image(path: Path) -> str:
    text = _extract_with_gemini(path)
    return text.strip()

OUTPUT_BASE_DIR = Path(tempfile.gettempdir()) if os.getenv("VERCEL") else BASE_DIR
EXTRACTED_TEXT_DIR = OUTPUT_BASE_DIR / "extracted_text"


def _build_output_txt_path(original_filename: str) -> Path:
    """Create deterministic output path: <original filename>.txt"""
    source_name = Path(original_filename).name
    txt_name = f"{source_name}.txt"
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    return EXTRACTED_TEXT_DIR / txt_name


def _append_page_block(path: Path, block: str) -> None:
    separator = ""
    if path.exists() and path.stat().st_size > 0:
        separator = "\n\n"
    with path.open("a", encoding="utf-8") as out_file:
        out_file.write(f"{separator}{block}")


def _write_full_text(path: Path, text: str) -> None:
    path.write_text(text or "", encoding="utf-8")


def _extract_pdf_page_text(page) -> str:
    words = page.get_text("words", sort=True) or []
    if not words:
        return ""

    lines: list[list[tuple[float, str]]] = []
    current_line: list[tuple[float, str]] = []
    current_y: float | None = None
    line_tolerance = 3.0

    for word in words:
        x0, y0, _x1, _y1, text = word[:5]
        if current_y is None or abs(y0 - current_y) <= line_tolerance:
            current_line.append((float(x0), str(text)))
            current_y = float(y0) if current_y is None else current_y
            continue

        lines.append(current_line)
        current_line = [(float(x0), str(text))]
        current_y = float(y0)

    if current_line:
        lines.append(current_line)

    page_lines: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        rendered = ""
        last_column = 0

        for x0, text in line:
            column = max(0, round(x0 / 4))
            spaces = max(1, column - last_column)
            rendered += (" " * spaces) + text if rendered else (" " * column) + text
            last_column = column + len(text)

        page_lines.append(rendered.rstrip())

    return "\n".join(line for line in page_lines if line.strip()).strip()


def extract_from_pdf(file_bytes: bytes, original_filename: str) -> str:

    try:
        import fitz

    except ImportError as exc:

        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. "
            "Install pymupdf."
        ) from exc

    # Open PDF from memory
    doc = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    total_pages = len(doc)
    output_txt_path = _build_output_txt_path(original_filename)
    if output_txt_path.exists():
        output_txt_path.unlink()

    full_text: list[str] = []

    try:

        for page_number in range(total_pages):

            actual_page = page_number + 1

            print(f"\n📄 Processing Page {actual_page}/{total_pages}")

            page = doc.load_page(page_number)

            # Extract by visual position so headings stay before tables and table columns remain readable.
            native_text = _extract_pdf_page_text(page)

            page_text = native_text

            # If no text found -> OCR image page
            if not page_text:

                print(
                    f"🖼 OCR image-based PDF page "
                    f"{actual_page}/{total_pages}"
                )

                pix = page.get_pixmap(alpha=False)

                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ) as tmp:

                    temp_path = Path(tmp.name)

                try:

                    pix.save(str(temp_path))

                    page_text = (
                        process_image(temp_path) or ""
                    ).strip()

                finally:

                    if temp_path.exists():
                        temp_path.unlink()

            formatted_text = (
                f"--- Page {actual_page} ---\n"
                f"{page_text or ''}"
            )

            _append_page_block(output_txt_path, formatted_text)
            full_text.append(formatted_text)

    finally:

        doc.close()

    # Final combined text
    final_text = "\n\n".join(full_text).strip()

    return final_text
def extract_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX extraction.") from exc

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = Path(tmp.name)
    try:
        document = Document(str(temp_path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs).strip()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def extract_from_doc(file_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        doc_path = Path(temp_dir) / "input.doc"
        txt_path = Path(temp_dir) / "output.txt"
        doc_path.write_bytes(file_bytes)

        try:
            import win32com.client  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pywin32 is required for .doc extraction on Windows.") from exc

        word = document = None
        try:
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            document = word.Documents.Open(str(doc_path))
            document.SaveAs(str(txt_path), FileFormat=7)
        finally:
            if document is not None:
                try:
                    document.Close(False)
                except Exception:
                    pass
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass

        for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                content = txt_path.read_text(encoding=encoding).strip()
                if content:
                    return content
            except UnicodeError:
                continue
        return txt_path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_from_excel(file_bytes: bytes) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for Excel extraction.") from exc

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = Path(tmp.name)
    try:
        workbook = load_workbook(str(temp_path), data_only=True)
        parts: list[str] = []
        for sheet in workbook.worksheets:
            parts.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/styles.css")
async def styles() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "styles.css"))


@app.get("/app.js")
async def app_script() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "app.js"))


@app.post("/api/extract")
async def extract_text(
    file: UploadFile = File(...)
) -> JSONResponse:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded."
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Empty file."
        )

    output_txt_path = _build_output_txt_path(file.filename)

    try:

        # =========================
        # IMAGE FILES
        # =========================
        if suffix in SUPPORTED_EXTENSIONS:

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as tmp:

                tmp.write(content)

                temp_path = Path(tmp.name)

            try:

                text = process_image(temp_path)

            finally:

                if temp_path.exists():
                    temp_path.unlink()

            _write_full_text(output_txt_path, text)

        # =========================
        # PDF FILES
        # =========================
        elif suffix in PDF_EXTENSIONS:

            text = extract_from_pdf(
                content,
                file.filename
            )

        # =========================
        # DOCX FILES
        # =========================
        elif suffix in WORD_EXTENSIONS:

            text = extract_from_docx(content)
            _write_full_text(output_txt_path, text)

        # =========================
        # DOC FILES
        # =========================
        elif suffix in LEGACY_WORD_EXTENSIONS:

            text = extract_from_doc(content)
            _write_full_text(output_txt_path, text)

        # =========================
        # EXCEL FILES
        # =========================
        elif suffix in EXCEL_EXTENSIONS:

            text = extract_from_excel(content)
            _write_full_text(output_txt_path, text)

        # =========================
        # UNSUPPORTED FILES
        # =========================
        else:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file. "
                    "Use image, PDF, DOC/DOCX, "
                    "or XLSX-family formats."
                ),
            )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {exc}"
        ) from exc

    return JSONResponse(
        {
            "filename": file.filename,
            "characters": len(text or ""),
            "text": text or "",
            "structured_rows": [],
            "table_html": [],
        }
    )
@app.post("/api/send-extracted")
async def send_extracted_text(payload: SendExtractedPayload) -> JSONResponse:
    try:
        response = requests.post(
            str(payload.endpoint),
            json={"text": payload.text},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send extracted text: {exc}") from exc

    response_data: dict | list | str
    try:
        response_data = response.json()
    except ValueError:
        response_data = response.text

    return JSONResponse(
        {
            "endpoint": str(payload.endpoint),
            "status_code": response.status_code,
            "response": response_data,
        }
    )


@app.get("/api/fetch-data")
async def fetch_data(endpoint: HttpUrl) -> JSONResponse:
    try:
        response = requests.get(str(endpoint), timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text}

    return JSONResponse(
        {
            "endpoint": str(endpoint),
            "status_code": response.status_code,
            "data": data,
        }
    )


@app.post("/api/process-extracted")
async def process_extracted(payload: ProcessExtractedPayload) -> JSONResponse:
    if not PROCESSOR_POST_ENDPOINT or not PROCESSOR_GET_ENDPOINT:
        raise HTTPException(
            status_code=500,
            detail="Missing PROCESSOR_POST_ENDPOINT or PROCESSOR_GET_ENDPOINT in environment.",
        )

    try:
        post_response = requests.post(
            PROCESSOR_POST_ENDPOINT,
            json={"text": payload.text},
            timeout=30,
        )
        post_response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed sending text to processor: {exc}") from exc

    last_error: Exception | None = None
    for _ in range(12):
        try:
            get_response = requests.get(PROCESSOR_GET_ENDPOINT, timeout=30)
            get_response.raise_for_status()
            try:
                processed_data = get_response.json()
            except ValueError:
                processed_data = {"raw": get_response.text}

            return JSONResponse(
                {
                    "post_endpoint": PROCESSOR_POST_ENDPOINT,
                    "get_endpoint": PROCESSOR_GET_ENDPOINT,
                    "processed_data": processed_data,
                }
            )
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1)

    raise HTTPException(
        status_code=504,
        detail=f"Processor did not return data in time: {last_error}",
    )


