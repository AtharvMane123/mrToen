# MarathiTranslator — offline desktop starter

Features:
- Marathi text → English translation
- Select DOCX or text-based PDF
- Translate extracted document text
- Save the result as a new DOCX using a Save As dialog
- Local inference; no translation API

## Important limitations
- First-time setup requires internet to download Python packages and the model.
- The selected Hugging Face model may require accepting its conditions on the model page before download.
- Scanned/image-only PDFs are not OCR'd by this starter. They need an OCR engine such as Tesseract/PaddleOCR added separately.
- This starter exports translated text to DOCX; it does not preserve the original PDF/DOCX layout, images, headers, footers, footnotes, or tables perfectly.
- Review translations before relying on them for legal or other high-stakes documents.
- Windows `.exe` should be built on Windows. The build output is a folder (`--onedir`) because PyTorch/model bundles are large. Keep the `models` directory accessible to the application.

## Ubuntu setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
python app.py
```

## Windows setup
1. Install Python 3.10 or 3.11 (64-bit).
2. Open Command Prompt in this folder.
3. Run:
```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_model.py
```
4. Test before packaging:
```bat
python app.py
```
5. Build:
```bat
build_windows.bat
```
6. Find the app under `dist\MarathiTranslator\MarathiTranslator.exe`.

## Model
IndicTrans2 Indic→English distilled 200M:
https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
