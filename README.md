# OCR

Universal Text Extractor is a local FastAPI web app for extracting text from images, PDFs, Word documents, and Excel files.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API keys, for example:

```env
GEMINI_API_KEY_1=your_api_key
```

## Run

```bash
uvicorn process_images:app --reload
```

Open `http://127.0.0.1:8000` in your browser.
