import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json

# ── Load model artifacts ──────────────────────────────────────────────────────
model   = joblib.load("fraud_model.joblib")
scaler  = joblib.load("scaler.joblib")

with open("feature_names.json") as f:
    FEATURE_NAMES = json.load(f)   # e.g. ['V1','V2',...,'V28','Amount']


def predict_csv(file):
    """
    Accepts a CSV file upload.
    Returns a styled results dataframe + a plain-English summary.
    """
    if file is None:
        return None, "⚠️ Please upload a CSV file."

    # ── 1. Read the file ──────────────────────────────────────────────────────
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"❌ Could not read file: {e}"

    # ── 2. Validate columns ───────────────────────────────────────────────────
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return None, (
            f"❌ Your CSV is missing these required columns: {missing}\n\n"
            f"Required columns are: {FEATURE_NAMES}"
        )

    # ── 3. Keep only the needed columns, in the right order ──────────────────
    X = df[FEATURE_NAMES].copy()

    # ── 4. Scale Amount the same way as training ──────────────────────────────
    X["Amount"] = scaler.transform(X[["Amount"]])

    # ── 5. Predict ────────────────────────────────────────────────────────────
    preds  = model.predict(X)
    probs  = model.predict_proba(X)[:, 1]   # probability of fraud

    # ── 6. Build results table ────────────────────────────────────────────────
    # Show original Amount (not scaled) for readability
    results = pd.DataFrame({
        "Transaction #": range(1, len(df) + 1),
        "Amount ($)":    df["Amount"].round(2),
        "Verdict":       ["🚨 FRAUD"     if p == 1 else "✅ Legitimate" for p in preds],
        "Fraud Risk":    [f"{prob*100:.1f}%" for prob in probs],
        "Confidence":    [f"{max(prob, 1-prob)*100:.1f}%" for prob in probs],
    })

    # ── 7. Summary stats ──────────────────────────────────────────────────────
    n_total    = len(df)
    n_fraud    = int(preds.sum())
    n_legit    = n_total - n_fraud
    fraud_pct  = n_fraud / n_total * 100
    avg_risk   = probs.mean() * 100

    if n_fraud == 0:
        alert = "🟢 All Clear"
        msg   = "No fraudulent transactions were detected."
    elif fraud_pct < 10:
        alert = "🟡 Low Alert"
        msg   = f"{n_fraud} suspicious transaction(s) flagged."
    elif fraud_pct < 30:
        alert = "🟠 Medium Alert"
        msg   = f"{n_fraud} transactions flagged — review recommended."
    else:
        alert = "🔴 High Alert"
        msg   = f"{n_fraud} transactions flagged — immediate review required!"

    summary = (
        f"## {alert}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total transactions analyzed | {n_total} |\n"
        f"| Flagged as FRAUD | {n_fraud} ({fraud_pct:.1f}%) |\n"
        f"| Marked Legitimate | {n_legit} |\n"
        f"| Average fraud risk score | {avg_risk:.1f}% |\n\n"
        f"**{msg}**"
    )

    return results, summary


# ── Build the Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(
    title="Bank Fraud Detector",
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
    css="""
    body { font-family: 'Segoe UI', sans-serif; }
    .header { text-align: center; padding: 20px 0 10px 0; }
    .header h1 { font-size: 2.2rem; margin-bottom: 4px; }
    .header p  { color: #888; font-size: 1rem; }
    .gr-button-primary { background: #c0392b !important; }
    """
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🏦 Bank Fraud Detector</h1>
        <p>Upload a CSV of transactions to scan for fraudulent activity</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Transactions")
            file_input = gr.File(
                label="Upload CSV file",
                file_types=[".csv"],
                type="filepath"
            )
            analyze_btn = gr.Button("🔍 Analyze Transactions", variant="primary", size="lg")

            gr.Markdown("""
            ### ℹ️ How to use
            1. Export your transaction data as a CSV
            2. Make sure it has the same columns as the training data (V1–V28 + Amount)
            3. Upload the file and click **Analyze**

            ### 📋 Expected columns
            `V1, V2, ..., V28, Amount`

            The model will predict whether each transaction is **Legitimate** or **Fraudulent**
            and give a **fraud risk percentage** for each row.
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            summary_output = gr.Markdown("*Results will appear here after analysis...*")
            table_output   = gr.Dataframe(
                label="Transaction-by-Transaction Results",
                wrap=True
            )

    analyze_btn.click(
        fn=predict_csv,
        inputs=[file_input],
        outputs=[table_output, summary_output]
    )

    gr.Markdown("""
    ---
    **Model:** Random Forest Classifier | **Dataset:** Credit Card Fraud Detection 2023 (Kaggle)
    | **Accuracy:** ~99% | *This tool is for educational purposes.*
    """)

if __name__ == "__main__":
    demo.launch()
