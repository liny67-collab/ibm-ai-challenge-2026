import os
import io
import re
import json
import base64
import tempfile
import difflib
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_DOCLING_CONVERTER = None
_RAPID_OCR_ENGINE = None


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        return None
    return OpenAI(api_key=api_key)


def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def get_rapidocr_engine():
    """Singleton loader for RapidOCR (ONNX runtime text box detection)"""
    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_ENGINE = RapidOCR()
        except ImportError:
            _RAPID_OCR_ENGINE = False
    return _RAPID_OCR_ENGINE if _RAPID_OCR_ENGINE is not False else None


def get_docling_converter():
    """Singleton loader for IBM Docling DocumentConverter with hardware acceleration"""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat, AcceleratorOptions, AcceleratorDevice

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = False
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO
        )

        _DOCLING_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    return _DOCLING_CONVERTER


def estimate_skew_angle(gray_img: Image.Image) -> float:
    """Fast projection-variance deskew angle estimator"""
    w, h = gray_img.size
    scale = 300.0 / max(w, 1)
    small = gray_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    arr = np.array(small, dtype=np.float32)
    inv = np.maximum(0, 255.0 - arr)
    inv[inv < 60] = 0

    best_angle = 0.0
    max_variance = -1.0
    for angle in np.arange(-8.0, 9.0, 2.0):
        rotated = Image.fromarray(inv).rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        var = float(np.var(np.sum(np.array(rotated), axis=1)))
        if var > max_variance:
            max_variance = var
            best_angle = angle

    return best_angle


def enhance_receipt_pil(file_bytes: bytes, target_width: int = 1200) -> Image.Image:
    """Receipt preprocessing: illumination background subtraction + auto-deskew"""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    gray = img.convert("L")

    w, h = gray.size
    if w > target_width:
        scale = target_width / float(w)
        gray = gray.resize((target_width, int(h * scale)), Image.Resampling.LANCZOS)

    bg = gray.filter(ImageFilter.GaussianBlur(radius=22))
    gray_arr = np.array(gray, dtype=np.float32)
    bg_arr = np.maximum(np.array(bg, dtype=np.float32), 1.0)
    norm_arr = np.clip((gray_arr / bg_arr) * 245.0, 0, 255).astype(np.uint8)
    enhanced = ImageOps.autocontrast(Image.fromarray(norm_arr), cutoff=2)

    angle = estimate_skew_angle(enhanced)
    if abs(angle) >= 1.5:
        enhanced = enhanced.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)

    return enhanced


def slice_tall_receipt(pil_img: Image.Image, slice_height: int = 900, overlap: int = 150) -> list[bytes]:
    """Overlapping sliding-window vertical slicer for tall thermal receipts"""
    w, h = pil_img.size
    if h <= w * 2.0:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        return [buf.getvalue()]

    slices = []
    step = slice_height - overlap
    for top in range(0, h, step):
        bottom = min(top + slice_height, h)
        crop = pil_img.crop((0, top, w, bottom))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        slices.append(buf.getvalue())
        if bottom >= h:
            break

    return slices


def group_boxes_into_rows(elements: list) -> list[str]:
    """Bounding-box Y-center row clustering to align left item names with right prices"""
    if not elements:
        return []

    elements.sort(key=lambda e: (e[0], e[1]))
    rows = []
    current_row = [elements[0]]

    for el in elements[1:]:
        avg_y = sum(item[0] for item in current_row) / len(current_row)
        avg_h = max(sum(item[2] for item in current_row) / len(current_row), 8.0)
        if abs(el[0] - avg_y) <= avg_h * 0.48:
            current_row.append(el)
        else:
            current_row.sort(key=lambda e: e[1])
            rows.append("   ".join(e[3] for e in current_row))
            current_row = [el]

    if current_row:
        current_row.sort(key=lambda e: e[1])
        rows.append("   ".join(e[3] for e in current_row))

    return rows


def extract_lines_by_coordinates(doc):
    """Extract bounding-box coordinates from IBM Docling document and cluster into rows"""
    elements = []
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        prov = getattr(item, "prov", None)
        if text and prov and len(prov) > 0:
            bbox = prov[0].bbox
            y_center = (bbox.t + bbox.b) / 2.0
            height = abs(bbox.t - bbox.b)
            x_left = bbox.l
            elements.append((y_center, x_left, height, text.strip()))

    if not elements:
        return doc.export_to_markdown().splitlines()
    return group_boxes_into_rows(elements)


def normalize_ocr_price_line(line: str) -> str:
    line = line.strip().strip("|").strip()
    line = re.sub(r"(\b\d{1,3})[,\s](\d{2})(?=\s|$|[A-Za-z])", r"\1.\2", line)
    return line.strip()


def normalize_keyword_text(text: str) -> str:
    trans = str.maketrans({"0": "O", "1": "I", "8": "B", "5": "S", "$": "S"})
    return text.upper().translate(trans)


def extract_receipt_total(raw_text_or_lines) -> float | None:
    """Extract final printed receipt total as a reconciliation benchmark"""
    lines = raw_text_or_lines if isinstance(raw_text_or_lines, list) else raw_text_or_lines.splitlines()
    total_candidates = []
    subtotal_candidates = []

    for raw_line in lines:
        clean_line = normalize_ocr_price_line(raw_line)
        norm_upper = normalize_keyword_text(clean_line)

        if any(ex in norm_upper for ex in ["SAVINGS", "SAVED", "DISCOUNT", "POINTS", "ITEMS", "TAX"]):
            continue

        m = re.search(
            r"\b(GRAND\s+TOTAL|TOTAL\s+PURCHASE|TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE|BALANCE)\b.*?(\d+\.\d{2})",
            norm_upper
        )
        if m:
            val = float(m.group(2))
            if 0 < val <= 10000:
                if "SUB" in norm_upper:
                    subtotal_candidates.append(val)
                else:
                    total_candidates.append(val)
            continue

        sub_m = re.search(r"\b(SUBTOTAL|SUB-TOTAL|SUB\s+TOTAL)\b.*?(\d+\.\d{2})", norm_upper)
        if sub_m:
            val = float(sub_m.group(2))
            if 0 < val <= 10000:
                subtotal_candidates.append(val)

    if total_candidates:
        return round(total_candidates[-1], 2)
    if subtotal_candidates:
        return round(subtotal_candidates[-1], 2)
    return None


SECTION_HEADERS = {
    "GROCERY", "HEALTH AND BEAUTY", "HEALTH & BEAUTY", "PRODUCE", "DAIRY",
    "HOME", "ELECTRONICS", "APPAREL", "MEAT", "FROZEN", "BAKERY", "DELI",
    "REFRIG/FROZEN", "BEVERAGES", "SNACKS", "HOUSEHOLD", "PERSONAL CARE",
    "PHARMACY", "BABY", "PETS", "SEASONAL", "STATIONERY", "TOYS"
}


def strip_section_prefix(text: str) -> str:
    """Strip bold category section headers (e.g., GROCERY) and leading SKU codes"""
    cleaned = text.strip()
    for sec in sorted(SECTION_HEADERS, key=len, reverse=True):
        pattern = rf"^{re.escape(sec)}(?:\s+|$)"
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
            break

    cleaned = re.sub(r"^\d{5,14}\s+", "", cleaned).strip()
    cleaned = re.sub(r"\b0([A-Za-z]{2,})", r"O\1", cleaned)
    return cleaned.strip(" -*#.")


def parse_bill_text(raw_text_or_lines):
    """Parse normalized receipt text lines into structured item dictionaries"""
    items = []
    ignore_keywords = [
        "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "TOTAL", "GRAND TOTAL",
        "TAX", "SALES TAX", "BALANCE", "AMOUNT DUE", "BALANCE DUE",
        "CHANGE", "CASH", "CREDIT", "DEBIT", "VISA", "MASTERCARD", "AMEX", "DISCOVER",
        "SAVINGS", "DISCOUNT", "COUPON", "MEMBER", "POINTS", "AMOUNT", "DUE",
        "ITEMS SOLD", "ITEM COUNT", "EFT", "APPROVED", "AUTH", "REF#", "AID",
        "TEND", "CARD", "ACCOUNT", "STORE", "THANK YOU", "CASHIER", "RECEIPT",
        "REGULAR PRICE", "YOU SAVED", "CLUB CARD", "REWARDS", "PRICE YOU PAY",
        "PURCHASE", "PAYMENT", "TIP", "GRATUITY", "DEBIT TEND", "VISA TEND",
        "TARGET CIRCLE", "RED CARD", "REDCARD"
    ]

    raw_lines = raw_text_or_lines if isinstance(raw_text_or_lines, list) else raw_text_or_lines.splitlines()
    pending_qty = 1
    pending_item_name = None

    for rl in raw_lines:
        clean_line = normalize_ocr_price_line(rl)
        if not clean_line or set(clean_line) <= {"-", "=", "_", " ", ":"}:
            continue

        if re.search(r"-\s*\$?\d+\.\d{2}|\d+\.\d{2}\s*-", clean_line):
            continue

        norm_upper = normalize_keyword_text(clean_line)
        if any(re.search(rf"\b{re.escape(kw)}\b", norm_upper) for kw in ignore_keywords):
            pending_item_name = None
            continue

        after_sec = strip_section_prefix(clean_line)
        if not after_sec or after_sec.upper() in SECTION_HEADERS:
            continue

        qty_only_match = re.match(r"^(\d+)\s*[@xX×]\s*\$?\d+\.\d{2}(?:\s*[A-Za-z\s]{1,5})?$", after_sec)
        if qty_only_match:
            pending_qty = int(qty_only_match.group(1))
            continue

        weight_match = re.search(
            r"(\d+\.\d+)\s*(?:lb|oz|kg)\s*@.*?\$?\s*(\d+\.\d{2})(?:\s+[A-Za-z0-9\s]{1,5})?$",
            after_sec,
            re.IGNORECASE
        )
        if weight_match:
            price = float(weight_match.group(2))
            if pending_item_name and 0 < price <= 3000:
                items.append({
                    "raw_name": pending_item_name,
                    "clean_name": f"{pending_item_name.title()} ({weight_match.group(1)} lb)",
                    "qty": 1,
                    "price": round(price, 2),
                    "suggest_aa": True,
                })
            pending_item_name = None
            pending_qty = 1
            continue

        match = re.search(
            r"^(?:(\d+)\s*[@xX×]\s*)?(.+?)\s+\$?\s*(\d+\.\d{2})(?:\s+[A-Za-z0-9]{1,3}){0,3}\s*$",
            after_sec
        )
        if match:
            inline_qty = match.group(1)
            raw_name = match.group(2).replace("|", " ").strip()
            price_str = match.group(3)

            qty = int(inline_qty) if inline_qty else pending_qty
            pending_qty = 1
            pending_item_name = None

            cleaned_name = strip_section_prefix(raw_name)
            cleaned_name = re.sub(r"\$\d+(?:\.\d{0,2})?", "", cleaned_name).strip()

            alpha_only = re.sub(r"[^A-Za-z]", "", cleaned_name)
            if len(alpha_only) < 3 or cleaned_name.upper() in SECTION_HEADERS:
                continue

            try:
                price = float(price_str)
                if price <= 0 or price > 3000:
                    continue
            except ValueError:
                continue

            display_name = f"{cleaned_name.title()} (x{qty})" if qty > 1 else cleaned_name.title()

            items.append({
                "raw_name": cleaned_name,
                "clean_name": display_name,
                "qty": qty,
                "price": round(price, 2),
                "suggest_aa": True,
            })
        else:
            upper_clean = after_sec.upper()
            if (
                len(re.findall(r"[A-Za-z]", after_sec)) >= 3
                and not re.search(r"\d+\.\d{2}", after_sec)
                and upper_clean not in SECTION_HEADERS
            ):
                pending_item_name = after_sec

    return items


def deduplicate_items(all_items: list, receipt_total: float | None = None) -> list:
    """Deduplicate overlapping slice items and filter out leaked receipt total lines"""
    deduped = []
    for cand in all_items:
        if receipt_total and receipt_total > 0 and abs(cand["price"] - receipt_total) <= 0.05:
            continue

        cand_alpha = re.sub(r"[^A-Z]", "", cand["raw_name"].upper())
        if len(cand_alpha) < 3:
            continue

        is_dup = False
        for prev in deduped:
            prev_alpha = re.sub(r"[^A-Z]", "", prev["raw_name"].upper())
            sim = difflib.SequenceMatcher(None, cand_alpha, prev_alpha).ratio()
            if sim >= 0.65 and abs(cand["price"] - prev["price"]) < 0.05:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
    return deduped


def run_rapidocr_on_slices(slices: list[bytes]) -> tuple[list, list[str]]:
    engine = get_rapidocr_engine()
    if engine is None:
        return [], []

    all_rows = []
    for s_bytes in slices:
        img = Image.open(io.BytesIO(s_bytes)).convert("RGB")
        result, _ = engine(np.array(img))
        if not result:
            continue

        elements = []
        for box, text, _conf in result:
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            y_center = sum(ys) / 4.0
            height = max(ys) - min(ys)
            x_left = min(xs)
            elements.append((y_center, x_left, height, text.strip()))

        all_rows.extend(group_boxes_into_rows(elements))

    return parse_bill_text(all_rows), all_rows


def run_tesseract_on_slices(slices: list[bytes]) -> tuple[list, list[str]]:
    import pytesseract

    all_lines = []
    for s_bytes in slices:
        img = Image.open(io.BytesIO(s_bytes))
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        all_lines.extend(text.splitlines())

    return parse_bill_text(all_lines), all_lines


def run_docling_on_slices(slices: list[bytes], file_ext: str = ".jpg") -> tuple[list, list[str]]:
    converter = get_docling_converter()
    all_rows = []

    for s_bytes in slices:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(s_bytes)
            tmp_path = tmp_file.name
        try:
            result = converter.convert(tmp_path)
            all_rows.extend(extract_lines_by_coordinates(result.document))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return parse_bill_text(all_rows), all_rows


def parse_receipt_with_total(file_bytes: bytes, filename: str):
    file_ext = os.path.splitext(filename)[1].lower() or ".jpg"
    is_pdf = (file_ext == ".pdf")
    client = get_openai_client()
    warning_msg = None

    if not is_pdf:
        enhanced_pil = enhance_receipt_pil(file_bytes, target_width=1200)
        slices = slice_tall_receipt(enhanced_pil)
    else:
        slices = [file_bytes]

    # 1. Optional Cloud Vision (GPT-4o)
    if client and not is_pdf:
        try:
            buf = io.BytesIO()
            enhanced_pil.save(buf, format="JPEG", quality=88)
            base64_img = encode_image(buf.getvalue())
            prompt = """
            Analyze this receipt image. Extract all purchased items AND the final printed receipt total.
            Return ONLY a valid JSON object with keys:
            - "items": array of objects with keys ("raw_name", "clean_name", "qty", "price", "suggest_aa")
            - "receipt_total": float number (the final Total / Amount Due printed at the bottom of the receipt, or null if not visible)
            Exclude subtotal, tax, total, category headers, and savings lines from "items".
            """
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=1500
            )
            content = re.sub(r"^```json\s*|\s*```$", "", response.choices[0].message.content.strip()).strip()
            data = json.loads(content)
            items = data.get("items", []) if isinstance(data, dict) else data
            printed_total = data.get("receipt_total") if isinstance(data, dict) else None
            if items:
                return items, "Cloud Vision (GPT-4o)", None, printed_total
        except Exception as e:
            err_str = str(e)
            warning_msg = (
                "OpenAI API quota exceeded (429). Automatically switched to the local multi-engine pipeline."
                if ("429" in err_str or "quota" in err_str.lower())
                else "Cloud vision model unavailable. Automatically switched to the local multi-engine pipeline."
            )
    elif not client and not is_pdf:
        warning_msg = f"Running local offline multi-engine extraction (receipt auto-segmented into {len(slices)} slice(s))."

    # 2. PDF Invoices -> IBM Docling
    if is_pdf:
        try:
            docling_items, raw_lines = run_docling_on_slices(slices, file_ext=".pdf")
            printed_total = extract_receipt_total(raw_lines)
            cleaned_items = deduplicate_items(docling_items, receipt_total=printed_total)
            if cleaned_items:
                return cleaned_items, "Local Document AI (IBM Docling)", warning_msg, printed_total
        except Exception as docling_err:
            print(f"[Docling PDF Fallback] {docling_err}")

    # 3. Receipt Images -> RapidOCR + Tesseract Fusion + IBM Docling Fallback
    if not is_pdf:
        rapid_items, rapid_lines = [], []
        tess_items, tess_lines = [], []
        try:
            rapid_items, rapid_lines = run_rapidocr_on_slices(slices)
        except Exception as rapid_err:
            print(f"[RapidOCR Engine] {rapid_err}")

        try:
            tess_items, tess_lines = run_tesseract_on_slices(slices)
        except Exception as tess_err:
            print(f"[Tesseract Engine] {tess_err}")

        all_raw_lines = rapid_lines + tess_lines
        printed_total = extract_receipt_total(all_raw_lines)

        base, extra = (rapid_items, tess_items) if len(rapid_items) >= len(tess_items) else (tess_items, rapid_items)
        merged_items = deduplicate_items(base + extra, receipt_total=printed_total)

        if len(merged_items) >= max(2, len(slices) * 2):
            engine_name = (
                f"Hybrid Local AI (RapidOCR + Tesseract + IBM Docling Ready, {len(slices)} Slice(s))"
                if rapid_items
                else f"Local Sliced OCR ({len(slices)} Slice(s))"
            )
            return merged_items, engine_name, warning_msg, printed_total

        docling_items, docling_lines = [], []
        try:
            docling_items, docling_lines = run_docling_on_slices(slices, file_ext=".jpg")
        except Exception as docling_err:
            print(f"[Docling Slice Fallback] {docling_err}")

        if printed_total is None:
            printed_total = extract_receipt_total(docling_lines)

        final_items = deduplicate_items(merged_items + docling_items, receipt_total=printed_total)
        if final_items:
            return final_items, f"IBM Docling + RapidOCR + Tesseract Fusion ({len(slices)} Slice(s))", warning_msg, printed_total

    fallback_notice = [{
        "raw_name": "MANUAL_INPUT_REQUIRED",
        "clean_name": "Could not extract valid price lines (please add items manually below)",
        "qty": 1,
        "price": 0.00,
        "suggest_aa": True,
    }]
    return fallback_notice, "Fallback Manual Mode", warning_msg or "No valid item lines detected. Please check image clarity.", None


def parse_receipt(file_bytes: bytes, filename: str):
    items, engine_used, warning_msg, _ = parse_receipt_with_total(file_bytes, filename)
    return items, engine_used, warning_msg


def parse_bill_image(file_bytes: bytes, filename: str = "receipt.jpg"):
    items, engine_used, warning_msg = parse_receipt(file_bytes, filename)
    return items, warning_msg


def parse_pdf_file(file_bytes: bytes, filename: str = "receipt.pdf"):
    items, engine_used, warning_msg = parse_receipt(file_bytes, filename)
    return items, warning_msg