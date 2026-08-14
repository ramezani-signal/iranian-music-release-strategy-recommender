from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_feature_matrix_v1.csv"
)

CONTRACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_feature_contract_v1.json"
)

FOLD_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nested_threshold_rf_fold_metrics_v1.csv"
)

PREDICTION_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nested_threshold_rf_oof_predictions_v1.csv"
)

THRESHOLD_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nested_threshold_rf_inner_audit_v1.csv"
)


RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 4

THRESHOLDS = np.round(
    np.arange(
        0.20,
        0.61,
        0.05,
    ),
    2,
)


def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(
    y_true,
    y_probability,
    threshold,
):
    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall_sensitivity":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    y_true,
    probability,
):
    rows = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_true,
            probability,
            threshold,
        )

        rows.append(
            {
                "threshold":
                    threshold,

                **metrics,
            }
        )

    audit = pd.DataFrame(rows)

    audit[
        "distance_from_default_threshold"
    ] = (
        audit["threshold"] - 0.50
    ).abs()

    selected = (
        audit.sort_values(
            [
                "balanced_accuracy",
                "f1",
                "recall_sensitivity",
                "distance_from_default_threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(selected["threshold"]),
        audit,
    )


def main():
    import json

    df = pd.read_csv(
        INPUT_FILE
    )

    with open(
        CONTRACT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        contract = json.load(file)

    model_features = contract[
        "model_features"
    ]

    target_column = "is_high_or_top"

    X = (
        df[model_features]
        .to_numpy(dtype=float)
    )

    y = (
        df[target_column]
        .to_numpy(dtype=int)
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    prediction_rows = []
    threshold_audit_rows = []

    for outer_fold, (
        train_index,
        test_index,
    ) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=(
                RANDOM_STATE
                + outer_fold
            ),
        )

        inner_probability = np.zeros(
            len(train_index),
            dtype=float,
        )

        for (
            inner_train_index,
            inner_validation_index,
        ) in inner_cv.split(
            X_train,
            y_train,
        ):
            model = build_model()

            model.fit(
                X_train[
                    inner_train_index
                ],
                y_train[
                    inner_train_index
                ],
            )

            inner_probability[
                inner_validation_index
            ] = model.predict_proba(
                X_train[
                    inner_validation_index
                ]
            )[:, 1]

        selected_threshold, threshold_audit = (
            select_threshold(
                y_train,
                inner_probability,
            )
        )

        threshold_audit[
            "outer_fold"
        ] = outer_fold

        threshold_audit[
            "selected_threshold"
        ] = (
            threshold_audit["threshold"]
            == selected_threshold
        ).astype(int)

        threshold_audit_rows.append(
            threshold_audit
        )

        final_model = build_model()

        final_model.fit(
            X_train,
            y_train,
        )

        test_probability = (
            final_model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_test,
            test_probability,
            selected_threshold,
        )

        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(
                y_test,
                test_probability,
            )

            average_precision = (
                average_precision_score(
                    y_test,
                    test_probability,
                )
            )
        else:
            roc_auc = np.nan
            average_precision = np.nan

        fold_rows.append(
            {
                "outer_fold":
                    outer_fold,

                "train_rows":
                    len(train_index),

                "test_rows":
                    len(test_index),

                "train_positive":
                    int(y_train.sum()),

                "test_positive":
                    int(y_test.sum()),

                "selected_threshold":
                    selected_threshold,

                **metrics,

                "roc_auc":
                    roc_auc,

                "average_precision":
                    average_precision,
            }
        )

        y_pred = (
            test_probability
            >= selected_threshold
        ).astype(int)

        fold_prediction = df.iloc[
            test_index
        ][
            [
                "record_id",
                "artist_name_fa",
                "category",
            ]
        ].copy()

        fold_prediction[
            "outer_fold"
        ] = outer_fold

        fold_prediction[
            "y_true"
        ] = y_test

        fold_prediction[
            "y_probability"
        ] = test_probability

        fold_prediction[
            "selected_threshold"
        ] = selected_threshold

        fold_prediction[
            "y_pred"
        ] = y_pred

        prediction_rows.append(
            fold_prediction
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    threshold_audit = pd.concat(
        threshold_audit_rows,
        ignore_index=True,
    )

    fold_metrics.to_csv(
        FOLD_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        PREDICTION_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    threshold_audit.to_csv(
        THRESHOLD_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pooled_metrics = calculate_metrics(
        predictions["y_true"].to_numpy(
            dtype=int
        ),
        predictions[
            "y_probability"
        ].to_numpy(dtype=float),
        threshold=0.50,
    )

    pooled_nested_pred = predictions[
        "y_pred"
    ].to_numpy(dtype=int)

    pooled_y = predictions[
        "y_true"
    ].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        pooled_y,
        pooled_nested_pred,
        labels=[0, 1],
    ).ravel()

    pooled_nested_specificity = (
        tn / (tn + fp)
    )

    pooled_nested = {
        "balanced_accuracy":
            balanced_accuracy_score(
                pooled_y,
                pooled_nested_pred,
            ),

        "precision":
            precision_score(
                pooled_y,
                pooled_nested_pred,
                zero_division=0,
            ),

        "recall_sensitivity":
            recall_score(
                pooled_y,
                pooled_nested_pred,
                zero_division=0,
            ),

        "specificity":
            pooled_nested_specificity,

        "f1":
            f1_score(
                pooled_y,
                pooled_nested_pred,
                zero_division=0,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    pooled_roc_auc = roc_auc_score(
        pooled_y,
        predictions[
            "y_probability"
        ],
    )

    pooled_average_precision = (
        average_precision_score(
            pooled_y,
            predictions[
                "y_probability"
            ],
        )
    )

    print(
        "\n=== NESTED THRESHOLD RANDOM FOREST V1 ===\n"
    )

    print(
        "Rows:",
        len(df),
    )

    print(
        "Model features:",
        len(model_features),
    )

    print(
        "Outer folds:",
        OUTER_SPLITS,
    )

    print(
        "Inner folds:",
        INNER_SPLITS,
    )

    print(
        "\n=== SELECTED THRESHOLD BY OUTER FOLD ===\n"
    )

    print(
        fold_metrics[
            [
                "outer_fold",
                "train_rows",
                "test_rows",
                "train_positive",
                "test_positive",
                "selected_threshold",
                "balanced_accuracy",
                "precision",
                "recall_sensitivity",
                "specificity",
                "f1",
                "roc_auc",
                "average_precision",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nThreshold distribution:"
    )

    print(
        fold_metrics[
            "selected_threshold"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nThreshold summary:"
    )

    print(
        fold_metrics[
            "selected_threshold"
        ]
        .describe()
        .round(4)
        .to_string()
    )

    print(
        "\n=== POOLED NESTED-THRESHOLD METRICS ===\n"
    )

    for key, value in pooled_nested.items():
        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"{key}: {value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        f"roc_auc: {pooled_roc_auc:.4f}"
    )

    print(
        "average_precision:",
        f"{pooled_average_precision:.4f}",
    )

    print(
        "\n=== SAME OOF PROBABILITIES AT "
        "FIXED THRESHOLD 0.50 ===\n"
    )

    for key, value in pooled_metrics.items():
        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"{key}: {value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nNested threshold predicted positive rate:"
    )

    print(
        f"{pooled_nested_pred.mean() * 100:.2f}%"
    )

    print(
        "\nObserved positive prevalence:"
    )

    print(
        f"{pooled_y.mean() * 100:.2f}%"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Thresholds were selected only from "
        "inner training-fold OOF predictions."
    )

    print(
        "Outer test folds were not used for "
        "threshold selection."
    )

    print(
        "This provides a leakage-controlled "
        "estimate of threshold tuning value."
    )

    print(
        "\nSaved fold metrics:"
    )
    print(FOLD_OUTPUT_FILE)

    print(
        "\nSaved OOF predictions:"
    )
    print(PREDICTION_OUTPUT_FILE)

    print(
        "\nSaved inner threshold audit:"
    )
    print(THRESHOLD_OUTPUT_FILE)


if __name__ == "__main__":
    main()
