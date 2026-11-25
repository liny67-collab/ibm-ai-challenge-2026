# 🧾 SmartSplit AI — Intelligent Receipt Parsing & Roommate Bill Splitter

> **Built for the IBM North America AI Builders Challenge**  
> A privacy-first, local-capable Document AI web application powered by **IBM Docling**, **RapidOCR**, and **Tesseract** that turns messy grocery receipts and PDF invoices into reconciled, ready-to-split roommate bills in seconds.

---

## 🎯 Problem Statement
Splitting shared grocery runs (e.g., Target, Albertsons, Costco) among roommates is tedious and error-prone:
- Long thermal receipts contain dozens of abbreviated item codes, multi-buy quantities (`2 @ $5.99`), weight-based pricing (`1.45 lb @ $2.99/lb`), section headers (`GROCERY`, `HEALTH & BEAUTY`), and sales tax.
- Manual entry into traditional bill-splitting apps takes 10–15 minutes per trip, while relying purely on cloud vision APIs introduces privacy concerns and recurring API costs.

## ✨ Key Features
1. **Multi-Engine Local Document AI Pipeline:**
   - **IBM Docling** for structured PDF invoice layout analysis and bounding-box coordinate extraction.
   - **RapidOCR (ONNX)** + **Tesseract PSM-6** with adaptive illumination equalization and auto-deskewing for fast, offline thermal receipt parsing.
   - Optional **OpenAI GPT-4o** cloud vision fallback when an API key is provided.
2. **Human-in-the-Loop Interactive Verification:**
   - Clean, editable data table (`Item Name`, `Qty`, `Subtotal`, `Split (AA)`) putting final split decisions directly in the user's hands.
3. **Real-Time Receipt Total Reconciliation & One-Click Tax Fill:**
   - Automatically extracts the receipt's printed `TOTAL` as a ground-truth benchmark, highlights any discrepancy against the sum of parsed items, and offers a one-click button to append the exact `Sales Tax` gap.
4. **Instant Payment Summary Generator:**
   - Dynamically calculates shared vs. personal totals and generates a copy-pasteable payment message for group chats (Venmo / Zelle / iMessage).

---

## 🏗️ System Architecture

```text
[PDF Invoice / Receipt Photo / Raw Text]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Image Preprocessing & Adaptive Slicing (parser.py)     │
│ • Illumination background subtraction & auto-deskew    │
│ • Sliding-window vertical slicing for tall receipts    │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Hybrid Extraction & Deduplication Engine               │
│ • IBM Docling (PDF & structured layout coordinates)    │
│ • RapidOCR + Tesseract PSM-6 (row Y-center clustering) │
│ • Section-header stripping & printed TOTAL extraction  │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Interactive Streamlit UI & Reconciliation (app.py)     │
│ • Live editable table + one-click Sales Tax gap fill   │
│ • Real-time AA pool calculation & settlement summary   │
└────────────────────────────────────────────────────────┘

## 🔬 Real-World Testing & Technical Feedback on IBM Docling

During end-to-end evaluation across digital PDFs and physical thermal receipts (Target & Albertsons), we observed the following behaviors:

* **Digital PDF Invoices (`100% Accuracy`)**: IBM Docling's layout analysis and bounding-box extraction reliably isolate item descriptions, quantities, prices, and sales tax with zero manual correction needed.
* **Short & Extra-Long Thermal Receipts (`Edge Cases & Mitigations`)**:
  * **Dual-Column & Curved Paper Drift**: Thermal rolls with dual right-aligned price columns (`Price` vs. `You Pay`) and slight paper curvature can introduce minor OCR typos in product names or one-line label shifts.
  * **Bottom Thermal Fading**: Standard open-source layout models (including IBM Docling's default pipelines) are primarily trained on standard A4 pages rather than extra-long thermal rolls, causing low-contrast fading near the receipt tail to drop trailing items.
  * **Engineering Takeaway**: Pairing aspect-ratio-aware sliding-window slicing with a **Human-in-the-Loop reconciliation table** (one-click `Split (AA)` checkbox denoising, inline cell editing, and quick-add tax/item gap fill) turns imperfect mobile receipt scans into verified roommate splits in seconds.

## 🚀 Quick Start

```bash
git clone [https://github.com/liny67-collab/ibm-ai-challenge-2026.git](https://github.com/liny67-collab/ibm-ai-challenge-2026.git)
cd ibm-ai-challenge-2026
pip install -r requirements.txt
streamlit run app.py
