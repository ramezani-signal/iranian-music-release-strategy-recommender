from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings(
    "ignore",
    category=UndefinedMetricWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATRIX_FILE = (
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
    / "baseline_model_fold_metrics_v1.csv"
)

SUMMARY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_model_summary_v1.csv"
)

PREDICTION_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_model_oof_predictions_v1.csv"
)


TARGET = "is_high_or_top"
GROUP_COLUMN = "artist_name_fa"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50


def safe_roc_auc(y_true, y_probability):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        y_probability,
    )


def safe_average_precision(
    y_true,
    y_probability,
):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(
        y_true,
        y_probability,
    )


def calculate_metrics(
    y_true,
    y_probability,
    threshold=DECISION_THRESHOLD,
):
    y_prediction = (
        np.asarray(y_probability)
        >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_prediction,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_prediction,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_prediction,
            ),

        "precision":
            precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),

        "recall_sensitivity":
            recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                y_prediction,
                zero_division=0,
            ),

        "roc_auc":
            safe_roc_auc(
                y_true,
                y_probability,
            ),

        "average_precision":
            safe_average_precision(
                y_true,
                y_probability,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_models():
    logistic_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_model = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    dummy_model = DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE,
    )

    return {
        "dummy_prior":
            dummy_model,

        "logistic_regression_balanced":
            logistic_model,

        "random_forest_balanced":
            random_forest_model,
    }


def evaluate_protocol(
    df,
    X,
    y,
    groups,
    model_name,
    model,
    protocol_name,
    splitter,
):
    fold_rows = []
    prediction_rows = []

    if protocol_name == "stratified_5_fold":
        split_iterator = splitter.split(
            X,
            y,
        )
    else:
        split_iterator = splitter.split(
            X,
            y,
            groups,
        )

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        split_iterator,
        start=1,
    ):
        estimator = clone(model)

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        estimator.fit(
            X_train,
            y_train,
        )

        probability = estimator.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            y_test.to_numpy(),
            probability,
        )

        test_artists = sorted(
            df.iloc[test_index][
                GROUP_COLUMN
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        held_out_artist = (
            test_artists[0]
            if (
                protocol_name
                == "leave_one_artist_out"
                and len(test_artists) == 1
            )
            else ""
        )

        fold_row = {
            "protocol":
                protocol_name,

            "model":
                model_name,

            "fold":
                fold_number,

            "held_out_artist":
                held_out_artist,

            "train_rows":
                len(train_index),

            "test_rows":
                len(test_index),

            "train_positive":
                int(y_train.sum()),

            "train_negative":
                int(
                    len(y_train)
                    - y_train.sum()
                ),

            "test_positive":
                int(y_test.sum()),

            "test_negative":
                int(
                    len(y_test)
                    - y_test.sum()
                ),

            **metrics,
        }

        fold_rows.append(fold_row)

        for local_position, row_index in enumerate(
            test_index
        ):
            prediction_rows.append(
                {
                    "protocol":
                        protocol_name,

                    "model":
                        model_name,

                    "fold":
                        fold_number,

                    "record_id":
                        df.iloc[row_index][
                            "record_id"
                        ],

                    "artist_name_fa":
                        df.iloc[row_index][
                            "artist_name_fa"
                        ],

                    "category":
                        df.iloc[row_index][
                            "category"
                        ],

                    "y_true":
                        int(
                            y_test.iloc[
                                local_position
                            ]
                        ),

                    "y_probability":
                        float(
                            probability[
                                local_position
                            ]
                        ),

                    "y_prediction":
                        int(
                            probability[
                                local_position
                            ]
                            >= DECISION_THRESHOLD
                        ),
                }
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
    )


def build_protocol_summary(
    fold_metrics,
    predictions,
):
    summary_rows = []

    grouped_keys = [
        "protocol",
        "model",
    ]

    for (
        protocol_name,
        model_name,
    ), prediction_group in predictions.groupby(
        grouped_keys
    ):
        fold_group = fold_metrics[
            (
                fold_metrics["protocol"]
                == protocol_name
            )
            & (
                fold_metrics["model"]
                == model_name
            )
        ]

        pooled_metrics = calculate_metrics(
            prediction_group[
                "y_true"
            ].to_numpy(),
            prediction_group[
                "y_probability"
            ].to_numpy(),
        )

        valid_fold_roc = (
            fold_group["roc_auc"]
            .dropna()
        )

        valid_fold_ap = (
            fold_group[
                "average_precision"
            ]
            .dropna()
        )

        summary_rows.append(
            {
                "protocol":
                    protocol_name,

                "model":
                    model_name,

                "rows":
                    len(prediction_group),

                "folds":
                    len(fold_group),

                "positive_rows":
                    int(
                        prediction_group[
                            "y_true"
                        ].sum()
                    ),

                "negative_rows":
                    int(
                        len(prediction_group)
                        - prediction_group[
                            "y_true"
                        ].sum()
                    ),

                "pooled_accuracy":
                    pooled_metrics[
                        "accuracy"
                    ],

                "pooled_balanced_accuracy":
                    pooled_metrics[
                        "balanced_accuracy"
                    ],

                "pooled_precision":
                    pooled_metrics[
                        "precision"
                    ],

                "pooled_recall_sensitivity":
                    pooled_metrics[
                        "recall_sensitivity"
                    ],

                "pooled_specificity":
                    pooled_metrics[
                        "specificity"
                    ],

                "pooled_f1":
                    pooled_metrics["f1"],

                "pooled_roc_auc":
                    pooled_metrics[
                        "roc_auc"
                    ],

                "pooled_average_precision":
                    pooled_metrics[
                        "average_precision"
                    ],

                "pooled_tn":
                    pooled_metrics["tn"],

                "pooled_fp":
                    pooled_metrics["fp"],

                "pooled_fn":
                    pooled_metrics["fn"],

                "pooled_tp":
                    pooled_metrics["tp"],

                "mean_fold_balanced_accuracy":
                    fold_group[
                        "balanced_accuracy"
                    ].mean(),

                "std_fold_balanced_accuracy":
                    fold_group[
                        "balanced_accuracy"
                    ].std(),

                "mean_fold_f1":
                    fold_group["f1"].mean(),

                "std_fold_f1":
                    fold_group["f1"].std(),

                "mean_valid_fold_roc_auc":
                    valid_fold_roc.mean(),

                "valid_roc_auc_fold_count":
                    len(valid_fold_roc),

                "mean_valid_fold_average_precision":
                    valid_fold_ap.mean(),

                "valid_average_precision_fold_count":
                    len(valid_fold_ap),
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    df = pd.read_csv(
        MATRIX_FILE
    )

    contract = json.loads(
        CONTRACT_FILE.read_text(
            encoding="utf-8"
        )
    )

    model_features = contract[
        "model_features"
    ]

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    if df[model_features].isna().sum().sum() != 0:
        raise ValueError(
            "Missing model feature values."
        )

    X = df[model_features].astype(float)
    y = df[TARGET].astype(int)
    groups = df[GROUP_COLUMN].astype(str)

    models = build_models()

    protocols = {
        "stratified_5_fold":
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),

        "leave_one_artist_out":
            LeaveOneGroupOut(),
    }

    all_fold_metrics = []
    all_predictions = []

    for protocol_name, splitter in (
        protocols.items()
    ):
        for model_name, model in (
            models.items()
        ):
            print(
                f"Evaluating {model_name} "
                f"with {protocol_name}..."
            )

            (
                fold_metrics,
                predictions,
            ) = evaluate_protocol(
                df=df,
                X=X,
                y=y,
                groups=groups,
                model_name=model_name,
                model=model,
                protocol_name=protocol_name,
                splitter=splitter,
            )

            all_fold_metrics.append(
                fold_metrics
            )

            all_predictions.append(
                predictions
            )

    fold_metrics = pd.concat(
        all_fold_metrics,
        ignore_index=True,
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = build_protocol_summary(
        fold_metrics,
        predictions,
    )

    FOLD_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== BASELINE MODEL SUMMARY V1 ===\n"
    )

    display_columns = [
        "protocol",
        "model",
        "pooled_balanced_accuracy",
        "pooled_precision",
        "pooled_recall_sensitivity",
        "pooled_specificity",
        "pooled_f1",
        "pooled_roc_auc",
        "pooled_average_precision",
        "pooled_tn",
        "pooled_fp",
        "pooled_fn",
        "pooled_tp",
        "mean_fold_balanced_accuracy",
        "std_fold_balanced_accuracy",
        "mean_valid_fold_roc_auc",
        "valid_roc_auc_fold_count",
    ]

    print(
        summary[
            display_columns
        ]
        .sort_values(
            [
                "protocol",
                "pooled_balanced_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== LEAVE-ONE-ARTIST-OUT "
        "FOLD DETAILS ===\n"
    )

    logo_details = fold_metrics[
        fold_metrics["protocol"]
        == "leave_one_artist_out"
    ][
        [
            "model",
            "fold",
            "held_out_artist",
            "test_rows",
            "test_positive",
            "test_negative",
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

    print(
        logo_details
        .sort_values(
            [
                "model",
                "fold",
            ]
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== BEST MODEL PER PROTOCOL "
        "BY BALANCED ACCURACY ===\n"
    )

    best_models = (
        summary.sort_values(
            [
                "protocol",
                "pooled_balanced_accuracy",
                "pooled_average_precision",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .groupby(
            "protocol",
            as_index=False,
        )
        .first()
    )

    print(
        best_models[
            [
                "protocol",
                "model",
                "pooled_balanced_accuracy",
                "pooled_f1",
                "pooled_roc_auc",
                "pooled_average_precision",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved fold metrics:")
    print(FOLD_OUTPUT_FILE)

    print("\nSaved OOF predictions:")
    print(PREDICTION_OUTPUT_FILE)

    print("\nSaved summary:")
    print(SUMMARY_OUTPUT_FILE)


if __name__ == "__main__":
    main()
