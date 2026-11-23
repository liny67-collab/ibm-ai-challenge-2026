# Core Logic 2: Dynamic Bill Splitting & Summary Generation

import pandas as pd

def calculate_split_bill(df: pd.DataFrame, num_roommates: int = 3, payer_name: str = "Alex") -> dict:
    """Calculate total expense, shared AA pool, per-person share, and generate summary text."""
    if df.empty:
        return {
            "total_expense": 0.0,
            "public_aa_total": 0.0,
            "per_person_aa_share": 0.0,
            "summary_text": "No data available"
        }

    total_expense = df["price"].sum()
    
    # Calculate total amount for items marked for AA splitting
    aa_df = df[df["suggest_aa"] == True]
    public_aa_total = aa_df["price"].sum()
    
    # Split evenly among all roommates (including the payer)
    per_person_aa_share = public_aa_total / num_roommates if num_roommates > 0 else 0.0
    
    # Extract list of shared AA item names
    aa_items_list = aa_df["clean_name"].tolist()
    items_str = ", ".join(aa_items_list) if aa_items_list else "None"

    summary_text = (
        f"🧾 [SmartSplit Bill Settlement]\n"
        f"Paid by: {payer_name}\n"
        f"Total Receipt Expense: ${total_expense:.2f}\n"
        f"------------------------------\n"
        f"Shared AA Items ({len(aa_items_list)}): {items_str}\n"
        f"Shared AA Pool Total: ${public_aa_total:.2f}\n"
        f"Total Roommates: {num_roommates}\n"
        f"👉 Each Roommate Owes {payer_name}: ${per_person_aa_share:.2f}\n"
        f"------------------------------\n"
        f"Thanks for splitting! Please send payment when convenient~ 🎉"
    )

    return {
        "total_expense": round(total_expense, 2),
        "public_aa_total": round(public_aa_total, 2),
        "per_person_aa_share": round(per_person_aa_share, 2),
        "summary_text": summary_text
    }