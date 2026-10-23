import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import threading
import tkinter as tk
import subprocess
import tempfile
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from translate import IndicTransONNX
from docx import Document
from pypdf import PdfReader
import fitz  # PyMuPDF: renders PDF pages for OCR fallback

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "hari31416/indictrans2-indic-en-dist-200M-ONNX"
SOURCE_LANGUAGE = "mar_Deva"
TARGET_LANGUAGE = "eng_Latn"
DEVICE = "cpu"

# Bundled Tesseract files expected beside app.py:
# tesseract/tesseract (Ubuntu) or tesseract/tesseract.exe (Windows)
TESSERACT_DIR = BASE_DIR / "tesseract"
TESSDATA_DIR = TESSERACT_DIR / "tessdata"
TESSERACT_EXE = TESSERACT_DIR / ("tesseract.exe" if os.name == "nt" else "tesseract")


def find_tesseract():
    """Use the project-bundled executable first, then a system installation."""
    if TESSERACT_EXE.is_file():
        return str(TESSERACT_EXE)
    from shutil import which
    exe = which("tesseract")
    if exe:
        return exe
    raise FileNotFoundError(
        "Tesseract was not found. Put its executable in the project's tesseract folder "
        "or install Tesseract on this computer."
    )


def ocr_image(image_path):
    """OCR an image using Marathi and English trained data."""
    executable = find_tesseract()
    command = [executable, str(image_path), "stdout", "-l", "mar+eng", "--psm", "3"]
    if TESSDATA_DIR.is_dir():
        command.extend(["--tessdata-dir", str(TESSDATA_DIR)])
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=180
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Tesseract OCR failed. Check that mar.traineddata and eng.traineddata exist "
            f"in {TESSDATA_DIR}. Details: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def looks_corrupted(text):
    """Detect empty text, replacement characters, and common private-use font mappings."""
    if not text or not text.strip():
        return True
    suspicious = 0
    visible = 0
    for ch in text:
        if ch.isspace():
            continue
        visible += 1
        cp = ord(ch)
        if (
            ch == "\ufffd"
            or 0xE000 <= cp <= 0xF8FF
            or 0xF0000 <= cp <= 0xFFFFD
            or 0x100000 <= cp <= 0x10FFFD
        ):
            suspicious += 1
    return visible == 0 or suspicious / max(visible, 1) > 0.02


class MarathiTranslator:
    """IndicTrans2 ONNX inference forced to CPU."""

    def __init__(self):
        print(f"Loading ONNX model {MODEL_NAME} on CPU...", flush=True)
        self.model = IndicTransONNX(
            MODEL_NAME,
            providers=["CPUExecutionProvider"],
        )
        print("ONNX model loaded successfully on CPU.", flush=True)

    def translate(self, text):
        text = text.strip()
        if not text:
            return ""
        result = self.model.translate(
            text,
            src_lang=SOURCE_LANGUAGE,
            tgt_lang=TARGET_LANGUAGE,
            max_new_tokens=128,
        )
        if isinstance(result, (list, tuple)):
            return str(result[0]) if result else ""
        return str(result)


def extract_docx(path):
    """Extract paragraphs and table rows from a DOCX in reading order (basic)."""
    doc = Document(str(path))
    blocks = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            blocks.append(("paragraph", paragraph.text))

    # Include table contents. This basic version exports table text as paragraphs.
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(value for value in cells if value)
            if line:
                blocks.append(("table", line))
    return blocks


def extract_pdf(path):
    """
    Extract valid selectable text. Render/OCR only pages with missing or corrupted text.
    Text is returned in bounded blocks so model input lengths stay manageable.
    """
    blocks = []
    reader = PdfReader(str(path))
    pdf_doc = fitz.open(str(path))

    try:
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if looks_corrupted(page_text):
                pdf_page = pdf_doc[page_number - 1]
                pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                with tempfile.TemporaryDirectory(prefix="mrtoen_ocr_") as temp_dir:
                    image_path = Path(temp_dir) / f"page_{page_number}.png"
                    pix.save(str(image_path))
                    page_text = ocr_image(image_path)

            if not page_text.strip():
                blocks.append(("paragraph", f"[Page {page_number}: OCR could not extract text.]"))
                continue

            # Keep natural line breaks, but don't make a single oversized model input.
            for line in page_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                while len(line) > 180:
                    cut = line.rfind(" ", 0, 181)
                    if cut < 60:
                        cut = 180
                    blocks.append(("paragraph", line[:cut].strip()))
                    line = line[cut:].strip()
                if line:
                    blocks.append(("paragraph", line))
    finally:
        pdf_doc.close()

    return blocks



def chunk_blocks(blocks, max_chars=180):
    """
    Combine neighboring short lines into chunks while ensuring each input remains
    within a conservative character limit for this model. Preserve table boundaries.
    """
    result = []
    current_kind = None
    current_parts = []
    current_len = 0

    def flush():
        nonlocal current_kind, current_parts, current_len
        if current_parts:
            result.append((current_kind or "paragraph", "\n".join(current_parts)))
        current_kind, current_parts, current_len = None, [], 0

    for kind, raw_text in blocks:
        text = (raw_text or "").strip()
        if not text:
            continue

        # Split single long blocks first.
        pieces = []
        while len(text) > max_chars:
            cut = text.rfind(" ", 0, max_chars + 1)
            if cut < max_chars // 3:
                cut = max_chars
            pieces.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            pieces.append(text)

        for piece in pieces:
            added_len = len(piece) + (1 if current_parts else 0)
            if current_parts and (kind != current_kind or current_len + added_len > max_chars):
                flush()
                added_len = len(piece)
            if not current_parts:
                current_kind = kind
            current_parts.append(piece)
            current_len += added_len

    flush()
    return result

def translate_document(engine, blocks, progress_callback=None):
    translated = []
    total = max(len(blocks), 1)
    for index, (kind, text) in enumerate(blocks, start=1):
        # Avoid sending empty strings to the model.
        result = engine.translate(text) if text.strip() else ""
        translated.append((kind, result))
        if progress_callback:
            progress_callback(index, total)
    return translated


def write_docx(blocks, destination):
    doc = Document()
    for kind, text in blocks:
        if not text.strip():
            doc.add_paragraph("")
        elif kind == "table":
            doc.add_paragraph(text)
        else:
            doc.add_paragraph(text)
    doc.save(str(destination))


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi to English Translator")
        self.root.geometry("900x720")
        self.root.minsize(700, 560)
        self.root.configure(bg="#f3f4f6")

        self.translator = None
        self.translator_ready = False
        self.busy = False
        self.selected_file = None
        self.translated_document = None

        self.build_ui()
        threading.Thread(target=self.load_model, daemon=True).start()

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text="Marathi → English Translator",
            font=("Arial", 21, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer, text="ONNX accelerated translation • CPU mode",
            foreground="#555"
        ).pack(anchor="w", pady=(2, 12))

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)

        # Text tab
        text_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(text_tab, text="Translate Text")

        ttk.Label(text_tab, text="Marathi text").pack(anchor="w")
        self.input_box = tk.Text(
            text_tab, height=10, wrap="word",
            font=("Noto Sans Devanagari", 12)
        )
        self.input_box.pack(fill="both", expand=True, pady=(5, 8))

        text_buttons = ttk.Frame(text_tab)
        text_buttons.pack(fill="x")
        self.text_translate_button = ttk.Button(
            text_buttons, text="Translate", command=self.start_text_translation,
            state="disabled"
        )
        self.text_translate_button.pack(side="left")
        ttk.Button(
            text_buttons, text="Clear", command=self.clear_text
        ).pack(side="left", padx=8)
        ttk.Button(
            text_buttons, text="Copy English", command=self.copy_text
        ).pack(side="left")

        ttk.Label(text_tab, text="English translation").pack(
            anchor="w", pady=(10, 0)
        )
        self.output_box = tk.Text(
            text_tab, height=10, wrap="word", font=("Arial", 11)
        )
        self.output_box.pack(fill="both", expand=True, pady=(5, 0))

        # Document tab: file picker and DOCX download/save
        file_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(file_tab, text="Translate DOCX / PDF")

        ttk.Label(
            file_tab,
            text="Select a Marathi Word document (.docx) or PDF (.pdf). Scanned/corrupted PDF pages use OCR:"
        ).pack(anchor="w")

        file_row = ttk.Frame(file_tab)
        file_row.pack(fill="x", pady=(8, 12))
        self.file_label = ttk.Label(
            file_row, text="No file selected", wraplength=600
        )
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(
            file_row, text="Browse…", command=self.pick_file
        ).pack(side="right")

        self.file_translate_button = ttk.Button(
            file_tab, text="Translate Document",
            command=self.start_document_translation, state="disabled"
        )
        self.file_translate_button.pack(anchor="w", pady=(0, 10))

        ttk.Label(file_tab, text="English translation preview").pack(anchor="w")
        self.file_preview = tk.Text(
            file_tab, height=18, wrap="word", font=("Arial", 10)
        )
        self.file_preview.pack(fill="both", expand=True, pady=(5, 10))

        self.download_button = ttk.Button(
            file_tab, text="Download Translation (.docx)",
            command=self.download_docx, state="disabled"
        )
        self.download_button.pack(anchor="w")

        self.status_var = tk.StringVar(value="Loading ONNX model on CPU…")
        ttk.Label(
            outer, textvariable=self.status_var, relief="sunken", anchor="w"
        ).pack(fill="x", pady=(10, 0))

    def load_model(self):
        try:
            engine = MarathiTranslator()
            self.root.after(0, lambda: self.on_model_ready(engine))
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self.on_model_error(error))

    def on_model_ready(self, engine):
        self.translator = engine
        self.translator_ready = True
        self.text_translate_button.configure(state="normal")
        self.file_translate_button.configure(state="normal")
        self.status_var.set("Ready — ONNX model loaded. Running on CPU.")

    def on_model_error(self, error):
        self.status_var.set("Could not load the model.")
        messagebox.showerror("Model loading error", error)

    def set_busy(self, busy, status):
        self.busy = busy
        self.status_var.set(status)
        button_state = "disabled" if busy or not self.translator_ready else "normal"
        self.text_translate_button.configure(state=button_state)
        self.file_translate_button.configure(state=button_state)
        self.download_button.configure(
            state="normal" if (not busy and self.translated_document) else "disabled"
        )

    def start_text_translation(self):
        if self.busy or not self.translator_ready:
            return
        text = self.input_box.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showinfo("No text", "Enter Marathi text first.")
            return
        self.set_busy(True, "Translating text with ONNX on CPU…")
        threading.Thread(
            target=self._text_worker, args=(text,), daemon=True
        ).start()

    def _text_worker(self, text):
        try:
            result = self.translator.translate(text)
            self.root.after(0, lambda: self._text_done(result))
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self.on_operation_error(error))

    def _text_done(self, result):
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", result)
        self.set_busy(False, "Text translation completed.")

    def clear_text(self):
        self.input_box.delete("1.0", "end")
        self.output_box.delete("1.0", "end")

    def copy_text(self):
        value = self.output_box.get("1.0", "end-1c").strip()
        if value:
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
            self.status_var.set("Translation copied to clipboard.")

    def pick_file(self):
        selected = filedialog.askopenfilename(
            title="Choose Marathi document",
            filetypes=[
                ("Supported documents", "*.docx *.pdf"),
                ("Word document", "*.docx"),
                ("PDF document", "*.pdf"),
            ],
        )
        if selected:
            self.selected_file = Path(selected)
            self.file_label.configure(text=str(self.selected_file))
            self.file_preview.delete("1.0", "end")
            self.translated_document = None
            self.download_button.configure(state="disabled")
            self.status_var.set(f"Selected: {self.selected_file.name}")

    def start_document_translation(self):
        if self.busy or not self.translator_ready:
            return
        if not self.selected_file:
            messagebox.showinfo("Choose a file", "Click Browse and select a DOCX or PDF.")
            return
        self.translated_document = None
        self.download_button.configure(state="disabled")
        self.set_busy(True, "Extracting document text…")
        threading.Thread(
            target=self._document_worker,
            args=(self.selected_file,),
            daemon=True,
        ).start()

    def _document_worker(self, path):
        try:
            if path.suffix.lower() == ".docx":
                source_blocks = extract_docx(path)
            elif path.suffix.lower() == ".pdf":
                source_blocks = extract_pdf(path)
            else:
                raise ValueError("Only .docx and .pdf files are supported.")

            if not source_blocks:
                raise ValueError("No text was found in the selected document.")

            safe_blocks = chunk_blocks(source_blocks, max_chars=180)
            if not safe_blocks:
                raise ValueError("No readable text was found in the selected document.")

            translated = []
            total = len(safe_blocks)
            for index, (kind, text) in enumerate(safe_blocks, start=1):
                result = self.translator.translate(text) if text.strip() else ""
                translated.append((kind, result))
                self.root.after(
                    0,
                    lambda i=index, n=total: self.status_var.set(
                        f"Translating document with ONNX on CPU… {i}/{n}"
                    ),
                )
            self.root.after(0, lambda: self._document_done(translated))
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self.on_operation_error(error))

    def _document_done(self, blocks):
        self.translated_document = blocks
        self.file_preview.delete("1.0", "end")
        self.file_preview.insert(
            "1.0", "\n\n".join(text for _, text in blocks)
        )
        self.set_busy(False, "Document translated. Click Download Translation (.docx).")
        self.download_button.configure(state="normal")

    def download_docx(self):
        if not self.translated_document:
            return
        stem = self.selected_file.stem if self.selected_file else "translation"
        destination = filedialog.asksaveasfilename(
            title="Save English translation",
            initialfile=f"{stem}_english.docx",
            defaultextension=".docx",
            filetypes=[("Word document", "*.docx")],
        )
        if not destination:
            return
        try:
            write_docx(self.translated_document, destination)
            self.status_var.set(f"Saved: {destination}")
            messagebox.showinfo(
                "Translation saved",
                f"The English DOCX was saved to:\n{destination}",
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def on_operation_error(self, error):
        self.set_busy(False, "Operation failed.")
        messagebox.showerror("Translation error", error)


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
