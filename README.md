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