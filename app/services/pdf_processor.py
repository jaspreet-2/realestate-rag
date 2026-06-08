import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
import uuid
import logging
from typing import List, Tuple, Dict
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

REAL_ESTATE_KEYWORDS = [
    # English
    "property", "land", "plot", "area", "sqft", "sq ft", "acres", "hectare",
    "bhk", "bedroom", "bathroom", "floor", "building", "apartment", "villa",
    "registry", "deed", "survey", "boundary", "price", "rate", "agreement",
    "sale", "purchase", "lease", "tenant", "landlord", "mortgage", "ownership",
    # Hindi
    "जमीन", "संपत्ति", "मकान", "प्लॉट", "क्षेत्रफल", "भूमि", "निर्माण",
    "बिक्री", "खरीद", "किराया", "मालिक", "पंजीकरण", "सर्वे", "सीमा",
    # Bengali
    "জমি", "সম্পত্তি", "বাড়ি", "প্লট", "জায়গা", "ভবন", "ফ্ল্যাট",
    "বিক্রয়", "ক্রয়", "ভাড়া", "মালিক", "নিবন্ধন",
    # Tamil
    "நிலம்", "சொத்து", "வீடு", "கட்டிடம்", "குத்தகை", "விற்பனை",
    "வாங்குதல்", "வரைபடம்", "எல்லை", "பதிவு",
    # Telugu
    "భూమి", "ఆస్తి", "ఇల్లు", "భవనం", "అద్దె", "అమ్మకం",
    "కొనుగోలు", "సర్వే", "హద్దు", "నమోదు",
    # Kannada
    "ಭೂಮಿ", "ಆಸ್ತಿ", "ಮನೆ", "ಕಟ್ಟಡ", "ಬಾಡಿಗೆ", "ಮಾರಾಟ",
    "ಖರೀದಿ", "ಸರ್ವೇ", "ಗಡಿ", "ನೋಂದಣಿ",
    # Malayalam
    "ഭൂമി", "സ്വത്ത്", "വീട്", "കെട്ടിടം", "വാടക", "വിൽപ്പന",
    "വാങ്ങൽ", "സർവ്വേ", "അതിർത്തി", "രജിസ്ട്രേഷൻ",
    # Gujarati
    "જમીન", "મિલકત", "મકાન", "ઇમારત", "ભાડું", "વેચાણ",
    "ખરીદી", "સર્વે", "સીમા", "નોંધણી",
    # Punjabi
    "ਜ਼ਮੀਨ", "ਜਾਇਦਾਦ", "ਘਰ", "ਇਮਾਰਤ", "ਕਿਰਾਇਆ", "ਵਿਕਰੀ",
    "ਖਰੀਦ", "ਸਰਵੇ", "ਸੀਮਾ", "ਰਜਿਸਟ੍ਰੇਸ਼ਨ",
    # Marathi
    "जमीन", "मालमत्ता", "घर", "इमारत", "भाडे", "विक्री",
    "खरेदी", "सर्वेक्षण", "सीमा", "नोंदणी",
    # Odia
    "ଜମି", "ସମ୍ପତ୍ତି", "ଘର", "ଅଟ୍ଟାଳିକା", "ଭଡ଼ା", "ବିକ୍ରୟ",
    # Urdu
    "زمین", "جائیداد", "مکان", "عمارت", "کرایہ", "فروخت",
]


def is_blur(image_array: np.ndarray, threshold: float = 80.0) -> bool:
    """Laplacian variance — below threshold means blurry."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def enhance_image(pil_image: Image.Image) -> Image.Image:
    """Increase contrast, sharpen, and convert to grayscale for better OCR."""
    img = pil_image.convert("L")                         # grayscale
    img = ImageEnhance.Contrast(img).enhance(2.5)        # boost contrast
    img = ImageEnhance.Sharpness(img).enhance(2.0)       # sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))   # remove noise
    # Adaptive thresholding via numpy
    arr = np.array(img)
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(arr)


def detect_language(text: str) -> str:
    """
    Detect language. Also uses Unicode script ranges as a fallback
    since langdetect sometimes misidentifies short Indian text.
    """
    # Script-range detection first (more reliable for Indian scripts)
    script = _detect_script(text)
    if script:
        return script
    try:
        if len(text.strip()) < 20:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


def _detect_script(text: str) -> str:
    """Identify Indian language by Unicode block of majority characters."""
    from collections import Counter
    script_chars = Counter()
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            script_chars["hi"] += 1   # Devanagari → Hindi/Marathi
        elif 0x0980 <= cp <= 0x09FF:
            script_chars["bn"] += 1   # Bengali
        elif 0x0B80 <= cp <= 0x0BFF:
            script_chars["ta"] += 1   # Tamil
        elif 0x0C00 <= cp <= 0x0C7F:
            script_chars["te"] += 1   # Telugu
        elif 0x0C80 <= cp <= 0x0CFF:
            script_chars["kn"] += 1   # Kannada
        elif 0x0D00 <= cp <= 0x0D7F:
            script_chars["ml"] += 1   # Malayalam
        elif 0x0A80 <= cp <= 0x0AFF:
            script_chars["gu"] += 1   # Gujarati
        elif 0x0A00 <= cp <= 0x0A7F:
            script_chars["pa"] += 1   # Punjabi (Gurmukhi)
        elif 0x0B00 <= cp <= 0x0B7F:
            script_chars["or"] += 1   # Odia
        elif 0x0600 <= cp <= 0x06FF:
            script_chars["ur"] += 1   # Urdu (Arabic script)
        elif 0x0D80 <= cp <= 0x0DFF:
            script_chars["si"] += 1   # Sinhala

    if not script_chars:
        return ""
    dominant, count = script_chars.most_common(1)[0]
    # Only trust if at least 10 script chars found
    return dominant if count >= 10 else ""


def get_tesseract_lang(detected: str) -> str:
    """Map detected language code to Tesseract language string."""
    mapping = {
        "hi": "hin+eng",          # Hindi
        "mr": "hin+eng",          # Marathi (Devanagari)
        "sa": "san+hin",          # Sanskrit
        "ne": "nep+hin",          # Nepali
        "bn": "ben+eng",          # Bengali
        "ta": "tam+eng",          # Tamil
        "te": "tel+eng",          # Telugu
        "kn": "kan+eng",          # Kannada
        "ml": "mal+eng",          # Malayalam
        "gu": "guj+eng",          # Gujarati
        "pa": "pan+eng",          # Punjabi
        "or": "ori+eng",          # Odia
        "ur": "urd+eng",          # Urdu
        "si": "sin+eng",          # Sinhala
        "en": "eng",              # English
    }
    return mapping.get(detected, "eng+hin")  # default: English + Hindi fallback


def pdf_page_to_image(page: fitz.Page, dpi: int = 200) -> Tuple[np.ndarray, Image.Image]:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes))
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img, pil_img


def extract_text_from_page(page: fitz.Page) -> Tuple[str, bool, bool]:
    """
    Returns (text, is_blurry, used_ocr).
    Falls back to OCR if native text is poor quality or page is blurry.
    """
    native_text = page.get_text("text").strip()

    # Good native text — use it
    if len(native_text) > 80 and not _looks_garbled(native_text):
        return native_text, False, False

    # Render to image and check blur
    cv_img, pil_img = pdf_page_to_image(page)
    blurry = is_blur(cv_img)

    if blurry:
        logger.info(f"Page {page.number + 1} is blurry — enhancing before OCR")
        pil_img = enhance_image(pil_img)
    else:
        pil_img = pil_img.convert("L")

    lang_hint = detect_language(native_text) if native_text else "en"
    tess_lang = get_tesseract_lang(lang_hint)

    try:
        ocr_text = pytesseract.image_to_string(
            pil_img,
            lang=tess_lang,
            config="--oem 3 --psm 6"
        ).strip()
        return ocr_text or native_text, blurry, True
    except Exception as e:
        logger.warning(f"OCR failed on page {page.number + 1}: {e}")
        return native_text, blurry, False


def _looks_garbled(text: str) -> bool:
    """Heuristic: high ratio of non-alphanumeric chars → garbled encoding."""
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    return alnum / max(len(text), 1) < 0.4


def has_diagrams_or_images(page: fitz.Page) -> bool:
    """Check if page contains embedded images or vector drawings."""
    if page.get_images(full=True):
        return True
    # Large number of drawing commands → likely a diagram
    drawings = page.get_drawings()
    return len(drawings) > 15


def classify_section(text: str, has_image: bool) -> str:
    if has_image and len(text.strip()) < 50:
        return "diagram"
    if "|" in text or "\t" in text or text.count("\n") > 5:
        if any(c.isdigit() for c in text):
            return "table"
    if any(kw in text.lower() for kw in REAL_ESTATE_KEYWORDS):
        return "text"
    return "mixed" if has_image else "text"


def validate_real_estate_pdf(pages_text: List[str]) -> Tuple[bool, str]:
    """Return (is_valid, reason). Checks if content is real-estate related."""
    combined = " ".join(pages_text[:5]).lower()
    hits = sum(1 for kw in REAL_ESTATE_KEYWORDS if kw in combined)
    if hits >= 2:
        return True, "ok"
    if len(combined.strip()) < 100:
        return False, "PDF appears to be empty or unreadable. Please upload a valid PDF."
    return False, (
        "This document does not appear to be a real estate document. "
        "Please upload property deeds, agreements, listings, or survey documents."
    )


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks on sentence/word boundaries.
    Handles sentence endings for all Indian languages."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Sentence endings across all supported Indian languages + English
            for sep in [
                ". ",     # English
                "। ",     # Hindi, Marathi, Sanskrit, Nepali (Devanagari danda)
                "।\n",
                "॥ ",     # Double danda (Sanskrit / verse)
                ".\n",
                "!\n", "?\n",
                "。",     # (just in case)
                "\n\n",
                "\n",
                " ",
            ]:
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def process_pdf(file_path: str, pdf_id: str, pdf_name: str,
                chunk_size: int = 600, overlap: int = 80) -> Dict:
    """
    Full pipeline: open → per-page extract → validate → chunk → build records.
    Returns dict with keys: records, languages, pages_with_images, warnings, valid, reason.
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    all_page_texts, languages, pages_with_images, warnings = [], set(), 0, []
    records = []

    for page_num in range(total_pages):
        page = doc[page_num]
        text, blurry, used_ocr = extract_text_from_page(page)
        has_img = has_diagrams_or_images(page)
        if has_img:
            pages_with_images += 1
        if blurry:
            warnings.append(f"Page {page_num + 1} was blurry — contrast enhanced before OCR.")

        lang = detect_language(text)
        languages.add(lang)
        all_page_texts.append(text)

        section_type = classify_section(text, has_img)
        chunks = chunk_text(text, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            records.append({
                "text": chunk,
                "metadata": {
                    "pdf_id": pdf_id,
                    "pdf_name": pdf_name,
                    "page_number": page_num + 1,
                    "chunk_index": idx,
                    "total_pages": total_pages,
                    "language": lang,
                    "has_image": has_img,
                    "section_type": section_type,
                    "char_count": len(chunk),
                    "source_path": file_path,
                }
            })

    doc.close()

    valid, reason = validate_real_estate_pdf(all_page_texts)
    return {
        "records": records,
        "languages": list(languages),
        "pages_with_images": pages_with_images,
        "warnings": warnings,
        "total_pages": total_pages,
        "valid": valid,
        "reason": reason,
    }
