import os
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
API_KEY = os.getenv("WINSTON_API_KEY")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def detect_ai_generated_text(text):
    url = "https://api.gowinston.ai/v2/ai-content-detection"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def analyze_pdf_for_ai_content(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {"error": "No text found in the PDF."}
    result = detect_ai_generated_text(text)
    return result 