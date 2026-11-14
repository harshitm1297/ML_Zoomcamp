"""Select, train, evaluate, and save the mutual-interest classifier."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from features import DATA_DIR, FEATURE_COLUMNS, build_dataset, model_input
from modeling import (
    choose_threshold,
    compare_models,
    make_model,
    out_of_fold_probability,
    score,
    split_by_event,
    split_summary,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model.bin"
DEFAULT_REPORT = PROJECT_DIR / "results" / "metrics.json"


def train(data_dir: Path, model_path: Path, report_path: Path) -> dict:
    frame = build_dataset(data_dir)
    train_frame, val_frame, test_frame = split_by_event(frame)
    development = pd.concat([train_frame, val_frame], ignore_index=True)
    comparisons = compare_models(development)
    winner = comparisons.iloc[0]
    name, params = str(winner["model"]), dict(winner["parameters"])
    validation_probability = out_of_fold_probability(development, name, params)
    threshold = choose_threshold(development.mutual_interest, validation_probability)
    final_model = make_model(name, params)
    final_model.fit(model_input(development), development.mutual_interest)
    test_probability = final_model.predict_proba(model_input(test_frame))[:, 1]

    report = {
        "problem": "pre-date prediction of mutual interest after a Mixmosa event date",
        "source_rows": len(frame),
        "positive_rate": float(frame.mutual_interest.mean()),
        "split": split_summary(train_frame, val_frame, test_frame),
        "selection_metric": "5-fold event-grouped out-of-fold average precision",
        "candidates": [
            {
                "model": row.model,
                "parameters": row.parameters,
                "roc_auc": row.roc_auc,
                "average_precision": row.average_precision,
            }
            for row in comparisons.itertuples(index=False)
        ],
        "selected_model": name,
        "selected_parameters": params,
        "threshold": threshold,
        "validation": score(development.mutual_interest, validation_probability, threshold),
        "test": score(test_frame.mutual_interest, test_probability, threshold),
        "feature_columns": list(FEATURE_COLUMNS),
        "preprocessing": "median imputation with missing indicators; standard scaling for logistic regression",
        "excluded": [
            "event and participant IDs", "both post-date decisions", "swiping and feedback",
            "ethnicity, religion, politics, income, drug use, and sexual history",
        ],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump({"model": final_model, "threshold": threshold}, f)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = train(args.data_dir, args.model, args.report)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "threshold": result["threshold"],
        "validation": result["validation"],
        "test": result["test"],
        "model_path": str(args.model),
        "report_path": str(args.report),
    }, indent=2))
