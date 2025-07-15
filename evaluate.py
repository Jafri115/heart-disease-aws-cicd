import argparse
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils import load_object


def load_data(path: str):
    """Load dataset and return features and target."""
    df = pd.read_csv(path)
    y = df["HeartDisease"]
    X = df.drop("HeartDisease", axis=1)
    return X, y


def evaluate(model, X, y):
    """Return basic classification metrics."""
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
    }


def main(model_path: str, preprocessor_path: str, data_path: str, output_path: str) -> None:
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)

    X, y = load_data(data_path)
    X_proc = preprocessor.transform(X)

    metrics = evaluate(model, X_proc, y)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", default="artifacts/model.pkl", help="Path to trained model pickle")
    parser.add_argument("--preprocessor", default="artifacts/preprocessor.pkl", help="Path to preprocessor pickle")
    parser.add_argument("--data", default="artifacts/test.csv", help="CSV file with evaluation data")
    parser.add_argument("--output", default="artifacts/eval_results.json", help="Where to save metrics JSON")
    args = parser.parse_args()

    main(args.model, args.preprocessor, args.data, args.output)
