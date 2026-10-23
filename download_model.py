from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "indictrans2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading IndicTrans2. Internet is required for this one-time setup.")
print("If Hugging Face asks you to accept model conditions, open the model page and accept them first.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR, safe_serialization=True)
print(f"Model saved to: {MODEL_DIR}")
