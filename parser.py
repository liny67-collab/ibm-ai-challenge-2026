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
    """Singleton loader for IBM Docling DocumentConverter with version-safe hardware acceleration"""
    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        # Digital PDFs already have embedded text layers; disabling redundant OCR prevents OCR typos
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False
        pdf_options.do_table_structure = False

        # Receipt images require OCR
        img_options = PdfPipelineOptions()
        img_options.do_ocr = True
        img_options.do_table_structure = False

        # Version-safe import for AcceleratorOptions (moved in newer docling releases)
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
            pdf_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
            img_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
        except ImportError:
            try:
                from docling.datamodel.base_models import AcceleratorOptions, AcceleratorDevice
                pdf_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
                img_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
            except ImportError:
                pass  # Fall back to Docling's built-in default accelerator settings

        _DOCLING_CONVERTER = DocumentConverter(
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=img_options),
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
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
    for angle in np.arange(-12.0, 13.0, 1.5):
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
    if abs(angle) >= 1.2:
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
    """
    Slope-compensated bounding-box row clustering:
    Aligns left item descriptions with right prices across curved/slanted thermal receipts.
    elements: list of (y_center, x_left, height, text)
    """
    if not elements:
        return []

    heights = sorted(max(e[2], 6.0) for e in elements)
    median_h = heights[len(heights) // 2]

    # Estimate global tilt slope (dy / dx) from pairs on roughly the same horizontal band
    slopes = []
    sorted_by_y = sorted(elements, key=lambda e: e[0])
    for i in range(len(sorted_by_y)):
        y1, x1, _, _ = sorted_by_y[i]
        for j in range(i + 1, min(i + 6, len(sorted_by_y))):
            y2, x2, _, _ = sorted_by_y[j]
            dy = y2 - y1
            dx = x2 - x1
            if abs(dy) <= median_h * 0.45 and abs(dx) >= 180:
                slopes.append(dy / dx)
    tilt_slope = float(np.median(slopes)) if len(slopes) >= 3 else 0.0

    # Deskew Y coordinates mathematically using detected line slope
    deskewed = []
    for y_c, x_l, h_val, txt in elements:
        y_corr = y_c - tilt_slope * x_l
        deskewed.append((y_corr, x_l, h_val, txt))

    deskewed.sort(key=lambda e: (e[0], e[1]))
    rows = []
    current_row = [deskewed[0]]

    for el in deskewed[1:]:
        avg_y = sum(item[0] for item in current_row) / len(current_row)
        if abs(el[0] - avg_y) <= median_h * 0.42:
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


def extract_receipt_amounts(raw_text_or_lines) -> tuple[float | None, set[float]]:
    """Extract final printed receipt total (including **** BALANCE) and subtotals"""
    lines = raw_text_or_lines if isinstance(raw_text_or_lines, list) else raw_text_or_lines.splitlines()
    total_candidates = []
    subtotal_candidates = []

    for raw_line in lines:
        clean_line = normalize_ocr_price_line(raw_line)
        norm_upper = normalize_keyword_text(clean_line)

        if any(ex in norm_upper for ex in ["SAVINGS", "SAVED", "DISCOUNT", "POINTS", "ITEMS", "TAX"]):
            continue

        sub_m = re.search(r"(?:SUBTOTAL|SUB-TOTAL|SUB\s+TOTAL).*?(\d+\.\d{2})", norm_upper)
        if sub_m:
            val = float(sub_m.group(1))
            if 0 < val <= 10000:
                subtotal_candidates.append(round(val, 2))
            continue

        m = re.search(
            r"(?:GRAND\s+TOTAL|TOTAL\s+PURCHASE|TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE|BALANCE).*?(\d+\.\d{2})",
            norm_upper
        )
        if m:
            val = float(m.group(1))
            if 0 < val <= 10000:
                total_candidates.append(round(val, 2))

    final_total = total_candidates[-1] if total_candidates else (subtotal_candidates[-1] if subtotal_candidates else None)
    return final_total, set(subtotal_candidates)


def extract_receipt_total(raw_text_or_lines) -> float | None:
    final_total, _ = extract_receipt_amounts(raw_text_or_lines)
    return final_total


SECTION_HEADERS = {
    "GROCERY", "HEALTH AND BEAUTY", "HEALTH & BEAUTY", "PRODUCE", "DAIRY",
    "HOME", "ELECTRONICS", "APPAREL", "MEAT", "FROZEN", "BAKERY", "DELI",
    "REFRIG/FROZEN", "BEVERAGES", "SNACKS", "HOUSEHOLD", "PERSONAL CARE",
    "PHARMACY", "BABY", "PETS", "SEASONAL", "STATIONERY", "TOYS", "MISCELLANEOUS"
}

TABLE_HEADER_WORDS = {
    "PRICE", "UNIT PRICE", "QTY", "QUANTITY", "DESCRIPTION", "ITEM", "ITEMS",
    "AMOUNT", "EXT PRICE", "YOU PAY", "PRICE YOU PAY", "YAU PAY", "PRICE YAU PAY"
}


def strip_section_prefix(text: str) -> str:
    """Strip bold category section headers (e.g., GROCERY) and leading SKU codes"""
    cleaned = text.strip()
    for sec in sorted(SECTION_HEADERS, key=len, reverse=True):
        pattern = rf"^{re.escape(sec)}(?:\s+|$)"
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
            break

    # Strip leading 5-14 digit SKU barcode (even if directly attached to letters or 2@)
    cleaned = re.sub(r"^\d{5,14}\s*", "", cleaned).strip()
    cleaned = re.sub(r"\b0([A-Za-z]{2,})", r"O\1", cleaned)
    return cleaned.strip(" -*#.")


def parse_bill_text(raw_text_or_lines):
    """
    Parse normalized receipt text lines into structured item dictionaries:
    - Rescues first-line prices that merged with table headers (including OCR variants like
      'Price You Pay' / 'Price Yau Pay' followed by 'NONG SHIM SHIN'), preventing row shift!
    - Strips SKU codes BEFORE detecting '2@' multi-buy prefixes.
    - Uses non-greedy name matching so the FIRST valid price on the line is captured.
    """
    items = []
    ignore_keywords = [
        "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "TOTAL", "GRAND TOTAL",
        "BALANCE", "AMOUNT DUE", "BALANCE DUE",
        "CHANGE", "CASH", "CREDIT", "DEBIT", "VISA", "MASTERCARD", "AMEX", "DISCOVER",
        "SAVINGS", "DISCOUNT", "COUPON", "MEMBER", "POINTS", "AMOUNT", "DUE",
        "ITEMS SOLD", "ITEM COUNT", "EFT", "APPROVED", "AUTH", "REF#", "AID",
        "TEND", "CARD", "ACCOUNT", "STORE", "THANK YOU", "CASHIER", "RECEIPT",
        "REGULAR PRICE", "YOU SAVED", "CLUB CARD", "REWARDS",
        "PURCHASE", "PAYMENT", "TIP", "GRATUITY", "DEBIT TEND", "VISA TEND",
        "TARGET CIRCLE", "RED CARD", "REDCARD", "TAXABLE", "TAX EXEMPT",
        "UNIT PRICE", "DESCRIPTION", "QTY", "QUANTITY", "SALE SAVINGS"
    ]

    raw_lines = raw_text_or_lines if isinstance(raw_text_or_lines, list) else raw_text_or_lines.splitlines()
    pending_qty = 1
    pending_item_name = None
    orphaned_header_price = None

    for rl in raw_lines:
        clean_line = normalize_ocr_price_line(rl)
        if not clean_line or set(clean_line) <= {"-", "=", "_", " ", ":"}:
            continue

        if re.search(r"-\s*\$?\d+\.\d{2}|\d+\.\d{2}\s*-", clean_line):
            continue

        # Detect if table header ('Price You Pay' / 'Price Yau Pay') swallowed the first item's price
        header_price_match = re.match(
            r"^(?:PRICE\s+Y[OA]U\s+PAY|PRICE|Y[OA]U\s+PAY)\s+\$?\s*(\d+\.\d{2})(?:\s+.*)?$",
            clean_line,
            re.IGNORECASE
        )
        if header_price_match:
            val = float(header_price_match.group(1))
            if 0 < val <= 3000:
                orphaned_header_price = round(val, 2)
            continue

        norm_upper = normalize_keyword_text(clean_line)
        if any(re.search(rf"\b{re.escape(kw)}\b", norm_upper) for kw in ignore_keywords):
            pending_item_name = None
            continue

        # Strip section headers & SKU barcodes FIRST so '5229404889 2@ ITEM' exposes '2@ ITEM'
        after_sec = strip_section_prefix(clean_line)
        if not after_sec or after_sec.upper() in SECTION_HEADERS or after_sec.upper() in TABLE_HEADER_WORDS:
            continue

        # Standalone price line (e.g., '4.29 4.29 S' or '13.99') when item name was on previous/next line
        lone_price_match = re.match(r"^\$?\s*(\d+\.\d{2})(?:\s+\$?\d+\.\d{2})?(?:\s+[A-Za-z0-9]{1,3}){0,2}\s*$", after_sec)
        if lone_price_match:
            val = float(lone_price_match.group(1))
            if 0 < val <= 3000:
                if pending_item_name:
                    cleaned_name = strip_section_prefix(pending_item_name)
                    display_name = f"{cleaned_name.title()} (x{pending_qty})" if pending_qty > 1 else cleaned_name.title()
                    items.append({
                        "raw_name": cleaned_name,
                        "clean_name": display_name,
                        "qty": pending_qty,
                        "price": round(val, 2),
                        "suggest_aa": True,
                    })
                    pending_item_name = None
                    pending_qty = 1
                elif orphaned_header_price is None:
                    orphaned_header_price = round(val, 2)
            continue

        # Standalone multiplier line like "2@ 10.99"
        qty_only_match = re.match(r"^(\d+)\s*[@xX×]\s*\$?\d+\.\d{2}(?:\s*[A-Za-z\s]{1,5})?$", after_sec)
        if qty_only_match:
            pending_qty = int(qty_only_match.group(1))
            continue

        # Weight line like "1.45 lb @ 5.99/lb 8.69"
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

        # Match optional "2@" quantity prefix, then NON-GREEDY item name up to the FIRST price (\d+\.\d{2})
        match = re.search(
            r"^(?:(\d+)\s*[@xX×]\s*)?([A-Za-z][A-Za-z0-9\s&\-'/\.\(\)]*?)\s+\$?\s*(\d+\.\d{2})(?:\s+.*)?$",
            after_sec
        )
        if match:
            inline_qty = match.group(1)
            raw_name = match.group(2).replace("|", " ").strip()
            price_str = match.group(3)

            qty = int(inline_qty) if inline_qty else pending_qty
            pending_qty = 1

            cleaned_name = strip_section_prefix(raw_name)
            cleaned_name = re.sub(r"\$\d+(?:\.\d{0,2})?", "", cleaned_name).strip()
            cleaned_name = re.sub(r"\s+\d+(?:\.\d+)?$", "", cleaned_name).strip()

            if cleaned_name.upper() in TABLE_HEADER_WORDS or cleaned_name.upper() in SECTION_HEADERS:
                continue

            alpha_only = re.sub(r"[^A-Za-z]", "", cleaned_name)
            if len(alpha_only) < 2:
                continue

            try:
                price = float(price_str)
                if price <= 0 or price > 3000:
                    continue
            except ValueError:
                continue

            # If a previous item had no price (e.g. NONG SHIM SHIN) and we saved the orphaned header price ($13.99),
            # flush that previous item FIRST so this line's price stays aligned with this line's item!
            if pending_item_name and orphaned_header_price is not None:
                prev_clean = strip_section_prefix(pending_item_name)
                if len(re.sub(r"[^A-Za-z]", "", prev_clean)) >= 2:
                    items.append({
                        "raw_name": prev_clean,
                        "clean_name": prev_clean.title(),
                        "qty": 1,
                        "price": orphaned_header_price,
                        "suggest_aa": True,
                    })
                pending_item_name = None
                orphaned_header_price = None

            is_tax_line = bool(re.search(r"\b(SALES\s+TAX|STATE\s+TAX|LOCAL\s+TAX|TAX)\b", cleaned_name, re.IGNORECASE))
            if is_tax_line:
                qty = 1
                display_name = "Sales Tax"
            else:
                display_name = f"{cleaned_name.title()} (x{qty})" if qty > 1 else cleaned_name.title()

            items.append({
                "raw_name": "SALES TAX" if is_tax_line else cleaned_name,
                "clean_name": display_name,
                "qty": qty,
                "price": round(price, 2),
                "suggest_aa": True,
            })
        else:
            upper_clean = after_sec.upper()
            if (
                len(re.findall(r"[A-Za-z]", after_sec)) >= 2
                and not re.search(r"\d+\.\d{2}", after_sec)
                and upper_clean not in SECTION_HEADERS
                and upper_clean not in TABLE_HEADER_WORDS
            ):
                if orphaned_header_price is not None:
                    qty_m = re.match(r"^(\d+)\s*[@xX×]\s*(.+)$", after_sec)
                    if qty_m:
                        q_val = int(qty_m.group(1))
                        nm = strip_section_prefix(qty_m.group(2))
                        d_nm = f"{nm.title()} (x{q_val})" if q_val > 1 else nm.title()
                    else:
                        q_val = 1
                        nm = strip_section_prefix(after_sec)
                        d_nm = nm.title()
                    items.append({
                        "raw_name": nm,
                        "clean_name": d_nm,
                        "qty": q_val,
                        "price": orphaned_header_price,
                        "suggest_aa": True,
                    })
                    orphaned_header_price = None
                    pending_item_name = None
                else:
                    pending_item_name = after_sec

    return items


def deduplicate_items(all_items: list, receipt_total: float | None = None, excluded_amounts: set[float] | None = None) -> list:
    """Deduplicate overlapping slice items and filter out leaked receipt total/subtotal lines"""
    deduped = []
    blocked = set(excluded_amounts or set())
    if receipt_total and receipt_total > 0:
        blocked.add(round(receipt_total, 2))

    for cand in all_items:
        if any(abs(cand["price"] - b) <= 0.03 for b in blocked):
            continue

        cand_alpha = re.sub(r"[^A-Z]", "", cand["raw_name"].upper())
        if len(cand_alpha) < 2:
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
            Analyze this receipt image. Extract all purchased items (including Sales Tax as an item if present) AND the final printed receipt total.
            Return ONLY a valid JSON object with keys:
- "items": array of objects with keys ("raw_name", "clean_name", "qty", "price", "suggest_aa")
- "receipt_total": float number (the final Total / Amount Due printed at the bottom of the receipt, or null if not visible)
            Exclude subtotal, total, category headers, and savings lines from "items".
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

    # 2. PDF Invoices -> IBM Docling (selecting the cleanest single extraction source)
    if is_pdf:
        docling_items, raw_lines = [], []
        pdf_items, pdf_lines = [], []
        docling_err_msg = None

        try:
            docling_items, raw_lines = run_docling_on_slices(slices, file_ext=".pdf")
        except Exception as docling_err:
            docling_err_msg = str(docling_err)
            print(f"[Docling PDF Fallback] {docling_err}")

        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            for page in pdf:
                textpage = page.get_textpage()
                pdf_lines.extend(textpage.get_text_range().splitlines())
            pdf_items = parse_bill_text(pdf_lines)
        except Exception as pdf_err:
            print(f"[PDF Text Layer] {pdf_err}")

        printed_total, subtotals = extract_receipt_amounts(pdf_lines + raw_lines)
        chosen_items = pdf_items if len(pdf_items) >= len(docling_items) else docling_items
        cleaned_items = deduplicate_items(chosen_items, receipt_total=printed_total, excluded_amounts=subtotals)

        if cleaned_items:
            return cleaned_items, "Local Document AI (IBM Docling)", warning_msg, printed_total

        if docling_err_msg:
            warning_msg = f"PDF Diagnostic: {docling_err_msg}"

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
        printed_total, subtotals = extract_receipt_amounts(all_raw_lines)

        base, extra = (rapid_items, tess_items) if len(rapid_items) >= len(tess_items) else (tess_items, rapid_items)
        merged_items = deduplicate_items(base + extra, receipt_total=printed_total, excluded_amounts=subtotals)

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
            printed_total, subtotals = extract_receipt_amounts(docling_lines)

        final_items = deduplicate_items(merged_items + docling_items, receipt_total=printed_total, excluded_amounts=subtotals)
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
