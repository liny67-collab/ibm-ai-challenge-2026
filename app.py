# Streamlit Main UI & Interactive Logic Entry Point

import streamlit as st
import pandas as pd
from parser import parse_receipt_with_total, parse_bill_text, extract_receipt_total
from calculator import calculate_split_bill

# 1. Page Configuration
st.set_page_config(page_title="SmartSplit AI", page_icon="🧾", layout="wide")

st.title("🧾 SmartSplit AI — Intelligent Receipt Parsing & Roommate Bill Splitter")
st.caption("Local PDF & Receipt Document AI · Multi-Engine Extraction · Real-Time Total Reconciliation · One-Click AA Split")

st.divider()

# 2. Initialize Session State
if "bill_df" not in st.session_state:
    st.session_state.bill_df = None
if "engine_used" not in st.session_state:
    st.session_state.engine_used = None
if "warning_msg" not in st.session_state:
    st.session_state.warning_msg = None
if "receipt_total" not in st.session_state:
    st.session_state.receipt_total = None
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0


def build_bill_dataframe(items: list, default_aa: bool) -> pd.DataFrame:
    df = pd.DataFrame(items)
    for col in ["clean_name", "qty", "price", "suggest_aa"]:
        if col not in df.columns:
            if col == "suggest_aa":
                df[col] = default_aa
            elif col == "price":
                df[col] = 0.0
            elif col == "qty":
                df[col] = 1
            else:
                df[col] = ""
    df["clean_name"] = df["clean_name"].fillna("").astype(str)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(1).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).astype(float)
    df["suggest_aa"] = df["suggest_aa"].fillna(default_aa).astype(bool)
    return df[["clean_name", "qty", "price", "suggest_aa"]].reset_index(drop=True)


# Sidebar: Global Settlement Parameters
with st.sidebar:
    st.header("⚙️ Split Settings")
    num_roommates = st.number_input("Total Roommates (Including You)", min_value=1, max_value=10, value=3, step=1)
    payer_name = st.text_input("Payer Name", value="Alex")
    default_aa = st.checkbox("Auto-check 'Split (AA)' for new items", value=True)
    
    st.divider()
    if st.button("🗑️ Clear Current Receipt", width="stretch"):
        st.session_state.bill_df = None
        st.session_state.engine_used = None
        st.session_state.warning_msg = None
        st.session_state.receipt_total = None
        st.session_state.editor_key += 1
        st.rerun()

# Two-Column Main Layout
col1, col2 = st.columns([1, 1.15])

# ==========================================
# Left Column: Document or Text Input
# ==========================================
with col1:
    st.header("1. Upload Receipt or Paste Text")
    tab1, tab2 = st.tabs(["📁 Upload Receipt (PDF / Image)", "📝 Paste Receipt Text"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose a local file (.pdf, .jpg, .jpeg, .png)", type=["pdf", "jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if file_type == "pdf":
                st.success(f"📄 Loaded PDF Invoice: {uploaded_file.name}")
                if st.button("🤖 Extract PDF Items with IBM Docling", type="primary", width="stretch"):
                    with st.spinner("Extracting structured line items via IBM Docling..."):
                        pdf_bytes = uploaded_file.getvalue()
                        items, engine_used, warning_msg, receipt_total = parse_receipt_with_total(pdf_bytes, uploaded_file.name)
                        if items:
                            for item in items:
                                item["suggest_aa"] = default_aa
                            st.session_state.bill_df = build_bill_dataframe(items, default_aa)
                            st.session_state.engine_used = engine_used
                            st.session_state.warning_msg = warning_msg
                            st.session_state.receipt_total = receipt_total
                            st.session_state.editor_key += 1
                            st.rerun()
                        else:
                            st.error("❌ Failed to extract items from the PDF. Please check the file format.")
            else:
                st.image(uploaded_file, caption="Uploaded Receipt Preview", width="stretch")
                if st.button("🚀 Parse Receipt Items", type="primary", width="stretch"):
                    with st.spinner("Extracting items, quantities, prices, and receipt total..."):
                        img_bytes = uploaded_file.getvalue()
                        items, engine_used, warning_msg, receipt_total = parse_receipt_with_total(img_bytes, uploaded_file.name)
                        if items:
                            for item in items:
                                item["suggest_aa"] = default_aa
                            st.session_state.bill_df = build_bill_dataframe(items, default_aa)
                            st.session_state.engine_used = engine_used
                            st.session_state.warning_msg = warning_msg
                            st.session_state.receipt_total = receipt_total
                            st.session_state.editor_key += 1
                            st.rerun()
                        else:
                            st.error("❌ Could not parse items from the image. Try adding items manually below.")

    with tab2:
        bill_input = st.text_area(
            "Paste raw receipt text:", 
            value="", 
            placeholder="Example:\n2@ VITA DRINK LEMON 11.98\n3@ POPCORNERS 13.47\n3@ SHRIMP COCKTAIL 30.00\nTOTAL 55.45",
            height=180
        )
        if st.button("🤖 Parse Text Receipt", type="primary", width="stretch"):
            if bill_input.strip():
                with st.spinner("Parsing text line items..."):
                    st.session_state.warning_msg = None
                    items = parse_bill_text(bill_input)
                    receipt_total = extract_receipt_total(bill_input)
                    if items:
                        for item in items:
                            item["suggest_aa"] = default_aa
                        st.session_state.bill_df = build_bill_dataframe(items, default_aa)
                        st.session_state.engine_used = "Local Rule & Regex Engine (Text Input)"
                        st.session_state.receipt_total = receipt_total
                        st.session_state.editor_key += 1
                        st.rerun()
                    else:
                        st.error("❌ No valid item-price lines found. Ensure each line ends with a price (e.g., 11.98).")
            else:
                st.warning("⚠️ Please paste or enter receipt text first!")

# ==========================================
# Right Column: Human-in-the-Loop Verification & AA Selection
# ==========================================
with col2:
    st.header("2. Verify Items & Select Split (AA)")

    if st.session_state.get("engine_used"):
        st.caption(f"⚙️ Active Extraction Engine: **{st.session_state.engine_used}**")
    if st.session_state.get("warning_msg"):
        st.info(st.session_state.warning_msg)

    if st.session_state.bill_df is not None and len(st.session_state.bill_df) > 0:
        edited_df = st.data_editor(
            st.session_state.bill_df,
            column_config={
                "clean_name": st.column_config.TextColumn("Item Name", width="medium"),
                "qty": st.column_config.NumberColumn("Qty", min_value=1, step=1, format="%d"),
                "price": st.column_config.NumberColumn("Subtotal ($)", format="$%.2f", min_value=0.0),
                "suggest_aa": st.column_config.CheckboxColumn("Split (AA)"),
            },
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"bill_editor_{st.session_state.editor_key}"
        )

        # Clean empty rows and enforce dtypes for instant recalculation
        clean_edited_df = edited_df.copy()
        clean_edited_df["price"] = pd.to_numeric(clean_edited_df["price"], errors="coerce").fillna(0.0)
        clean_edited_df["qty"] = pd.to_numeric(clean_edited_df["qty"], errors="coerce").fillna(1).astype(int)
        clean_edited_df["suggest_aa"] = clean_edited_df["suggest_aa"].fillna(False).astype(bool)
        clean_edited_df["clean_name"] = clean_edited_df["clean_name"].fillna("Unnamed Item").astype(str)
        clean_edited_df = clean_edited_df[
            ~((clean_edited_df["clean_name"].str.strip() == "") & (clean_edited_df["price"] == 0.0))
        ].reset_index(drop=True)
    else:
        clean_edited_df = None
        st.info("👈 Upload a PDF/image receipt or paste text on the left to verify line items and select AA splits.")

    # ==========================================
    # Quick Add Missing Items / One-Click Sales Tax Fill
    # ==========================================
    with st.expander("➕ Missing an Item or Sales Tax? Quick Add Here", expanded=False):
        printed_total_preview = st.session_state.get("receipt_total")
        if clean_edited_df is not None and printed_total_preview is not None:
            cur_checked_sum = round(float(clean_edited_df.loc[clean_edited_df["suggest_aa"], "price"].sum()), 2)
            tax_gap = round(printed_total_preview - cur_checked_sum, 2)
            if tax_gap > 0.01:
                if st.button(f"🧾 Auto-Add Difference as Sales Tax (+${tax_gap:.2f})", width="stretch"):
                    new_row = pd.DataFrame([{
                        "clean_name": "Sales Tax",
                        "qty": 1,
                        "price": tax_gap,
                        "suggest_aa": True,
                    }])
                    st.session_state.bill_df = pd.concat([clean_edited_df, new_row], ignore_index=True)
                    st.session_state.editor_key += 1
                    st.rerun()

        with st.form("manual_add_item_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns([2.2, 0.8, 1.0])
            new_name = f_col1.text_input("Item Name", placeholder="e.g., Organic Milk or Sales Tax")
            new_qty = f_col2.number_input("Qty", min_value=1, max_value=99, value=1, step=1)
            new_price = f_col3.number_input("Subtotal ($)", min_value=0.01, max_value=5000.0, value=5.00, step=0.50, format="%.2f")
            
            f_col4, f_col5 = st.columns([1.2, 2.8])
            auto_aa = f_col4.checkbox("Split (AA)", value=default_aa)
            submitted = f_col5.form_submit_button("✅ Add to Receipt Table", width="stretch")

            if submitted:
                if new_name.strip():
                    display_name = f"{new_name.strip().title()} (x{new_qty})" if new_qty > 1 else new_name.strip().title()
                    new_row = pd.DataFrame([{
                        "clean_name": display_name,
                        "qty": int(new_qty),
                        "price": round(float(new_price), 2),
                        "suggest_aa": auto_aa,
                    }])
                    base_df = clean_edited_df if clean_edited_df is not None else pd.DataFrame(columns=["clean_name", "qty", "price", "suggest_aa"])
                    st.session_state.bill_df = pd.concat([base_df, new_row], ignore_index=True)
                    if not st.session_state.engine_used:
                        st.session_state.engine_used = "Manual Input"
                    st.session_state.editor_key += 1
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter an item name before adding.")

    # ==========================================
    # Real-Time Reconciliation & Dynamic Settlement
    # ==========================================
    if clean_edited_df is not None and len(clean_edited_df) > 0:
        checked_sum = round(float(clean_edited_df.loc[clean_edited_df["suggest_aa"], "price"].sum()), 2)
        all_rows_sum = round(float(clean_edited_df["price"].sum()), 2)
        items_sum = checked_sum
        public_aa_sum = checked_sum
        per_person_share = round(public_aa_sum / max(int(num_roommates), 1), 2)

        calculate_split_bill(clean_edited_df, num_roommates=num_roommates, payer_name=payer_name)
        printed_total = st.session_state.get("receipt_total")

        st.divider()
        st.header("3. Real-Time Reconciliation & Summary")

        if printed_total is not None and printed_total > 0:
            diff = round(printed_total - items_sum, 2)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric(
                "Receipt Total",
                f"${printed_total:.2f}",
                help="Final total extracted directly from the bottom of the receipt as a reconciliation benchmark"
            )
            m_col2.metric(
                "Current Items Sum",
                f"${items_sum:.2f}",
                delta=f"Diff ${diff:+.2f}" if abs(diff) >= 0.01 else "Exact Match",
                delta_color="off" if abs(diff) < 0.01 else "inverse",
                help="Live sum of currently checked items in the table above (updates immediately when checkboxes or prices change)"
            )
            m_col3.metric(
                "Shared AA Pool",
                f"${public_aa_sum:.2f}",
                help="Total amount to be split equally among roommates"
            )
            m_col4.metric("Per Person (AA)", f"${per_person_share:.2f}")

            if abs(diff) < 0.01:
                st.success("✅ **Reconciled:** The sum of checked line items matches the receipt's printed total!")
            elif diff > 0:
                st.warning(f"⚠️ **Reconciliation Notice:** Checked items sum is **${diff:.2f}** less than the receipt total (check remaining valid items, or use the quick-add panel above to add missing items/tax).")
            else:
                st.warning(f"⚠️ **Reconciliation Notice:** Checked items sum exceeds the receipt total by **${abs(diff):.2f}** (uncheck duplicate/misread rows or adjust prices above).")
        else:
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Current Items Sum", f"${items_sum:.2f}")
            m_col2.metric("Shared AA Pool", f"${public_aa_sum:.2f}")
            m_col3.metric("Per Person (AA)", f"${per_person_share:.2f}")

        aa_items = clean_edited_df[clean_edited_df["suggest_aa"]]
        non_aa_items = clean_edited_df[~clean_edited_df["suggest_aa"]]
        summary_lines = [
            f"🧾 SmartSplit Bill Summary (Paid by: {payer_name})",
            f"----------------------------------------",
            f"• All Parsed Rows Total: ${all_rows_sum:.2f}",
            f"• Checked / Shared AA Pool ({len(aa_items)} items): ${public_aa_sum:.2f}",
            f"• Unchecked / Excluded Rows ({len(non_aa_items)} items): ${round(all_rows_sum - public_aa_sum, 2):.2f}",
            f"• Split Among: {num_roommates} roommates",
            f"👉 Amount Each Roommate Owes {payer_name}: ${per_person_share:.2f}",
        ]
        st.subheader("📋 Copyable Payment Request")
        st.code("\n".join(summary_lines), language="text")
