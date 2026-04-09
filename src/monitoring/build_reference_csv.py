"""Usage:
python -m src.monitoring.build_reference_csv
"""

from src.monitoring.monitor_pro import load_s3_predictions

df = load_s3_predictions("predictions/train_ref")
df.to_csv("data/processed/train_with_preds.csv", index=False)
print("train_with_preds.csv créé")


def main():
    from src.monitoring.monitor_pro import load_s3_predictions

    df = load_s3_predictions("predictions/train_ref")
    df.to_csv("data/processed/train_with_preds.csv", index=False)
    print("train_with_preds.csv créé")


if __name__ == "__main__":
    main()
