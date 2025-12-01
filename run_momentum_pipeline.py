import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import statsmodels.api as sm

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentiment_signals import groupby_date, normalize_senti_score, generate_signals


MODEL_DIR = "results"
DATASET_CSV = "dataset.csv"
PRICE_CSV = "^GSPC.csv"


# ================================================
# 1. Load trained model (MPS if available)
# ================================================

def load_model_and_tokenizer():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ================================================
# 2. Batch prediction
# ================================================

def batch_predict(df, text_col="processed", batch_size=64):
    tokenizer, model, device = load_model_and_tokenizer()

    texts = df[text_col].astype(str).tolist()
    preds = []

    print(f"Running prediction on {len(texts)} tweets ...")

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(batch_preds)

    return np.array(preds)


def numeric_to_label(pred_id):
    mapping = {0: "bearish", 1: "bullish"}
    return mapping.get(pred_id, "bearish")


# ================================================
# 3. Build daily sentiment signals
# ================================================

def build_daily_signal_from_predictions(pred_csv="predicted_tweet_sentiment.csv"):
    df = pd.read_csv(pred_csv)
    df["senti_label"] = df["pred_label"].apply(numeric_to_label)

    df_date = groupby_date(df)
    df_date = normalize_senti_score(df_date)
    df_date = generate_signals(df_date)

    return df_date


# ================================================
# 4. Load price and compute returns
# ================================================

def load_price_data(path=PRICE_CSV):
    sp = pd.read_csv(path)
    sp = sp.rename(columns={"Date": "date"})
    return sp


def compute_returns_with_signal(df_date, sp):
    merged = pd.merge(df_date, sp, on="date", how="inner").copy()

    merged["Ret"] = merged["Adj Close"] / merged["Adj Close"].shift(1) - 1
    merged["ModelRet"] = merged["signal"].shift(1) * merged["Ret"]

    merged = merged.dropna(subset=["Ret", "ModelRet"])
    return merged


# ================================================
# 5. Core 3 metrics (annual return, Sharpe, max drawdown)
# ================================================

def compute_core_metrics(merged):
    ret = merged["ModelRet"].dropna()

    # Annualized return
    annual_ret = (1 + ret).prod() ** (252 / len(ret)) - 1

    # Sharpe ratio
    sharpe = ret.mean() / ret.std() if ret.std() != 0 else np.nan

    # Max Drawdown
    cum = (1 + ret).cumprod()
    run_max = cum.cummax()
    drawdown = cum / run_max - 1
    max_dd = drawdown.min()

    return annual_ret, sharpe, max_dd


# ================================================
# 6. Plot strategy vs benchmark + metrics box
# ================================================

def plot_strategy_vs_benchmark(merged, annual_ret, sharpe, max_dd,
                               title="Sentiment Momentum Strategy vs S&P500"):

    # Normalized horizontal axis: 0 to 1
    N = len(merged)
    dates = np.linspace(0, 1, N)  # normalized timeline

    bench_cum = 100 * (1 + merged["Ret"]).cumprod()
    strat_cum = 100 * (1 + merged["ModelRet"]).cumprod()

    run_max = strat_cum.cummax()
    drawdown = strat_cum / run_max - 1.0

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # -------- Top panel: cumulative return --------
    ax1.plot(dates, bench_cum, label="Benchmark (S&P 500)", linewidth=2.0)
    ax1.plot(dates, strat_cum, label="Sentiment Momentum Strategy", linewidth=2.2)

    ax1.set_title(title, fontsize=16, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (start = 100)")
    ax1.legend(loc="upper left", fontsize=11)

    # -------- Bottom panel: drawdown --------
    ax2.fill_between(dates, drawdown, 0, color="tab:red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Normalized Time (0 → 1)")
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks(ax2.get_yticks())
    ax2.set_yticklabels([f"{x*100:.0f}%" for x in ax2.get_yticks()])

    fig.tight_layout()
    plt.show()



# ================================================
# 7. Main pipeline
# ================================================

def main():
    df = pd.read_csv(DATASET_CSV)
    if "processed" not in df.columns or "date" not in df.columns:
        raise ValueError("dataset.csv must contain 'processed' and 'date' columns.")

    preds = batch_predict(df, text_col="processed", batch_size=64)
    df_pred = df.copy()
    df_pred["pred_label"] = preds
    df_pred.to_csv("predicted_tweet_sentiment.csv", index=False)
    print("Saved predictions to predicted_tweet_sentiment.csv")

    df_date = build_daily_signal_from_predictions("predicted_tweet_sentiment.csv")
    sp = load_price_data(PRICE_CSV)
    merged = compute_returns_with_signal(df_date, sp)

    annual_ret, sharpe, max_dd = compute_core_metrics(merged)
    print(f"Annualized Return: {annual_ret:.4f}")
    print(f"Daily Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.4f}")

    plot_strategy_vs_benchmark(merged, annual_ret, sharpe, max_dd)


if __name__ == "__main__":
    main()
