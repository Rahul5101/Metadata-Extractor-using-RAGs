import pandas as pd
import json

# Define the fields to evaluate
FIELDS = [
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
]

def normalize(val):
    if isinstance(val, str):
        return val.strip().lower()
    return str(val).strip().lower()

def calculate_recall(ground_truth_df, predictions):
    field_scores = {field: {"true": 0, "false": 0} for field in FIELDS}

    for idx, row in ground_truth_df.iterrows():
        doc_id = row["file"] if "file" in row else f"doc_{idx}"
        prediction = predictions[idx] if idx < len(predictions) else {}

        for field in FIELDS:
            expected = normalize(row.get(field, ""))
            predicted = normalize(prediction.get(field, ""))

            if expected and expected == predicted:
                field_scores[field]["true"] += 1
            else:
                field_scores[field]["false"] += 1

    recalls = {
        field: round(
            field_scores[field]["true"] / max(1, (field_scores[field]["true"] + field_scores[field]["false"])), 3
        ) for field in FIELDS
    }

    return recalls

if __name__ == "__main__":
    gt_path = "data/test.csv"
    pred_path = "predictions.json"

    gt_df = pd.read_csv(gt_path)
    with open(pred_path, "r") as f:
        predictions = json.load(f)

    recalls = calculate_recall(gt_df, predictions)

    print("\nPer-field Recall Scores:")
    for field, score in recalls.items():
        print(f"{field}: {score}")
