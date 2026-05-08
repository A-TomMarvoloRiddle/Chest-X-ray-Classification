import os
import csv

# ===== CONFIG =====
TEST_DIR = "dataset/test"
PRED_FILE = "predictions.csv"

VALID_LABELS = {"normal", "pneumonia"}


def load_ground_truth(test_dir):
    gt = {}
    for label in os.listdir(test_dir):
        class_path = os.path.join(test_dir, label)
        if not os.path.isdir(class_path):
            continue

        for img in os.listdir(class_path):
            gt[img] = label

    return gt


def load_predictions(pred_file):
    preds = {}

    with open(pred_file, "r") as f:
        reader = csv.DictReader(f)

        legacy_columns = {"image_name", "label"}
        detailed_columns = {"Filename", "True_Label", "Predicted_Label"}

        if legacy_columns.issubset(reader.fieldnames):
            for row in reader:
                img = row["image_name"].strip()
                label = row["label"].strip().lower()
                preds[img] = label
            return preds

        if detailed_columns.issubset(reader.fieldnames):
            for row in reader:
                img = row["Filename"].strip()
                label = row["Predicted_Label"].strip().lower()
                preds[img] = label
            return preds

        raise ValueError(
            "CSV must contain either 'image_name,label' or "
            "'Filename,True_Label,Predicted_Label' columns"
        )

    return preds


def evaluate(gt, preds):
    missing = set(gt.keys()) - set(preds.keys())
    extra = set(preds.keys()) - set(gt.keys())

    invalid = [k for k, v in preds.items() if v not in VALID_LABELS]

    correct = 0
    for img, true_label in gt.items():
        if preds.get(img) == true_label:
            correct += 1

    total = len(gt)
    accuracy = correct / total if total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
    }


def main():
    gt = load_ground_truth(TEST_DIR)
    preds = load_predictions(PRED_FILE)

    results = evaluate(gt, preds)

    print("\n===== RESULTS =====")
    print(f"Total images: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    print("\n===== ISSUES =====")
    print(f"Missing predictions: {len(results['missing'])}")
    print(f"Extra predictions: {len(results['extra'])}")
    print(f"Invalid labels: {len(results['invalid'])}")

    if results["missing"]:
        print("Sample missing:", list(results["missing"])[:5])
    if results["extra"]:
        print("Sample extra:", list(results["extra"])[:5])
    if results["invalid"]:
        print("Sample invalid:", results["invalid"][:5])


if __name__ == "__main__":
    main()
