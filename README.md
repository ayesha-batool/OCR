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

## Deploy on Render

This app is configured for [Render](https://render.com). Connect the GitHub repo and Render will use `render.yaml`:

- **Build:** `pip install -r requirements.txt`
- **Start:** `uvicorn process_images:app --host 0.0.0.0 --port $PORT`

Set environment variables in the Render dashboard (same as local `.env`):

```env
GEMINI_API_KEY_10=your_key
GEMINI_API_KEY_11=your_key
CORS_ORIGINS=https://your-frontend.onrender.com
```

On Render, extracted text is written to `/tmp` (ephemeral). For local runs, files are saved under `extracted_text/`.
