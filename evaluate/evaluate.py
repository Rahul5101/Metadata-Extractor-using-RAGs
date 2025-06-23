import pandas as pd
import json

METADATA_FIELDS = [
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
]


Mapping = {
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}


def normalize(value):
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def per_field_recall(ground_truth_df, prediction_data):
    truth_value = {field: {"true": 0, "false": 0} for field in METADATA_FIELDS}

    for index, row in ground_truth_df.iterrows():
        pred = prediction_data[index] if index < len(prediction_data) else {}

        for field in METADATA_FIELDS:
            expected = normalize(row.get(Mapping[field], ""))
            predicted = normalize(pred.get(field, ""))

            if expected and expected == predicted:
                truth_value[field]["true"] += 1
            else:
                truth_value[field]["false"] += 1

    recall_scores = {
        field: round(
            truth_value[field]["true"] / max(1, (truth_value[field]["true"] + truth_value[field]["false"])), 3
        )
        for field in METADATA_FIELDS
    }

    return recall_scores


if __name__ == "__main__":
    ground_truth_path = "data/test.csv"
    predictions_path = "predictions.json"


    ground_truth_df = pd.read_csv(ground_truth_path)
    with open(predictions_path, "r") as f:
        prediction_data = json.load(f)


    per_field_recall = per_field_recall(ground_truth_df, prediction_data)

    print("\n Per-field Recall Scores:")
    for field, score in per_field_recall.items():
        print(f"{field}: {score}")
