from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_model_oof_predictions_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "threshold_sensitivity_audit_v1.csv"
)


PRIMARY_PROTOCOL = "stratified_5_fold"
PRIMARY_MODEL = "random_forest_balanced"


THRESHOLDS = np.round(
    np.arange(
        0.20,
        0.81,
        0.05,
    ),
    2,
)


def calculate_threshold_metrics(
    y_true,
    probability,
    threshold,
):
    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    predicted_positive_rate = (
        prediction.mean()
    )

    return {
        "threshold":
            threshold,

        "accuracy":
            accuracy_score(
                y_true,
                prediction,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction,
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "recall_sensitivity":
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "negative_predictive_value":
            negative_predictive_value,

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "predicted_positive_rate":
            predicted_positive_rate,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    predictions = pd.read_csv(
        PREDICTION_FILE
    )

    subset = predictions[
        (
            predictions["protocol"]
            == PRIMARY_PROTOCOL
        )
        & (
            predictions["model"]
            == PRIMARY_MODEL
        )
    ].copy()

    if len(subset) != 115:
        raise ValueError(
            "Expected 115 OOF predictions "
            f"for primary model, found {len(subset)}."
        )

    if subset["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate OOF record IDs detected."
        )

    if subset[
        [
            "y_true",
            "y_probability",
        ]
    ].isna().sum().sum() != 0:
        raise ValueError(
            "Missing OOF prediction values."
        )

    y_true = (
        subset["y_true"]
        .to_numpy(dtype=int)
    )

    probability = (
        subset["y_probability"]
        .to_numpy(dtype=float)
    )

    rows = []

    for threshold in THRESHOLDS:
        rows.append(
            calculate_threshold_metrics(
                y_true,
                probability,
                threshold,
            )
        )

    audit = pd.DataFrame(rows)

    audit[
        "distance_from_default_threshold"
    ] = (
        audit["threshold"] - 0.50
    ).abs()

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== THRESHOLD SENSITIVITY AUDIT V1 ===\n"
    )

    print(
        "Protocol:",
        PRIMARY_PROTOCOL,
    )

    print(
        "Model:",
        PRIMARY_MODEL,
    )

    print(
        "Rows:",
        len(subset),
    )

    print(
        "Observed positive prevalence:",
        f"{y_true.mean() * 100:.2f}%",
    )

    print(
        "\n=== ALL THRESHOLDS ===\n"
    )

    display_columns = [
        "threshold",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "predicted_positive_rate",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print(
        audit[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    best_balanced = (
        audit.sort_values(
            [
                "balanced_accuracy",
                "f1",
                "distance_from_default_threshold",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    best_f1 = (
        audit.sort_values(
            [
                "f1",
                "balanced_accuracy",
                "distance_from_default_threshold",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    recall_candidates = audit[
        audit["recall_sensitivity"]
        >= 0.70
    ]

    if not recall_candidates.empty:
        best_recall_constrained = (
            recall_candidates
            .sort_values(
                [
                    "specificity",
                    "precision",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )
    else:
        best_recall_constrained = None

    print(
        "\n=== BEST THRESHOLD BY "
        "BALANCED ACCURACY ===\n"
    )

    print(
        best_balanced[
            display_columns
        ]
        .round(4)
        .to_string()
    )

    print(
        "\n=== BEST THRESHOLD BY F1 ===\n"
    )

    print(
        best_f1[
            display_columns
        ]
        .round(4)
        .to_string()
    )

    print(
        "\n=== BEST SPECIFICITY WITH "
        "RECALL >= 0.70 ===\n"
    )

    if best_recall_constrained is None:
        print(
            "No threshold satisfies recall >= 0.70."
        )
    else:
        print(
            best_recall_constrained[
                display_columns
            ]
            .round(4)
            .to_string()
        )

    default_row = audit[
        audit["threshold"] == 0.50
    ]

    print(
        "\n=== DEFAULT THRESHOLD 0.50 ===\n"
    )

    print(
        default_row[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "This audit uses pooled OOF predictions "
        "and is exploratory."
    )

    print(
        "A final optimized threshold must be "
        "selected inside training folds "
        "using nested cross-validation."
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
