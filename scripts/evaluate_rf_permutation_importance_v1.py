from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


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
    / "rf_permutation_importance_fold_v1.csv"
)

SUMMARY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rf_permutation_importance_summary_v1.csv"
)

FAMILY_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rf_permutation_importance_family_v1.csv"
)


TARGET = "is_high_or_top"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 50


FEATURE_FAMILIES = {
    "strategy_rz_duration_minutes":
        "duration",

    "strategy_rz_release_hour_tehran":
        "timing",

    "strategy_rz_days_since_previous_release_imputed":
        "release_cadence",

    "strategy_rz_title_character_count":
        "title_structure",

    "strategy_rz_title_word_count":
        "title_structure",

    "is_iran_weekend_release":
        "timing",

    "days_since_previous_release_missing":
        "release_cadence",

    "title_has_official":
        "title_format",

    "title_has_music_video":
        "title_format",

    "title_has_lyric_video":
        "title_format",

    "title_has_collaboration":
        "collaboration",

    "title_contains_persian":
        "title_language",

    "title_is_bilingual":
        "title_language",

    "category_classic_fusion":
        "category_context",

    "category_pop":
        "category_context",

    "category_rap_hiphop":
        "category_context",
}


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


def main():
    df = pd.read_csv(MATRIX_FILE)

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

    unmapped_features = sorted(
        set(model_features)
        - set(FEATURE_FAMILIES)
    )

    if unmapped_features:
        raise ValueError(
            "Features without family mapping: "
            f"{unmapped_features}"
        )

    X = df[model_features].astype(float)
    y = df[TARGET].astype(int)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    fold_performance_rows = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        test_probability = model.predict_proba(
            X_test
        )[:, 1]

        fold_roc_auc = roc_auc_score(
            y_test,
            test_probability,
        )

        fold_average_precision = (
            average_precision_score(
                y_test,
                test_probability,
            )
        )

        fold_performance_rows.append(
            {
                "fold":
                    fold_number,

                "test_rows":
                    len(test_index),

                "test_positive":
                    int(y_test.sum()),

                "roc_auc":
                    fold_roc_auc,

                "average_precision":
                    fold_average_precision,
            }
        )

        importance_ap = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="average_precision",
            n_repeats=N_REPEATS,
            random_state=(
                RANDOM_STATE
                + fold_number
            ),
            n_jobs=-1,
        )

        importance_roc = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=N_REPEATS,
            random_state=(
                RANDOM_STATE
                + 100
                + fold_number
            ),
            n_jobs=-1,
        )

        impurity_importance = (
            model.feature_importances_
        )

        for feature_index, feature in enumerate(
            model_features
        ):
            fold_rows.append(
                {
                    "fold":
                        fold_number,

                    "feature":
                        feature,

                    "feature_family":
                        FEATURE_FAMILIES[
                            feature
                        ],

                    "test_rows":
                        len(test_index),

                    "test_positive":
                        int(y_test.sum()),

                    "fold_roc_auc":
                        fold_roc_auc,

                    "fold_average_precision":
                        fold_average_precision,

                    "permutation_ap_mean":
                        importance_ap.importances_mean[
                            feature_index
                        ],

                    "permutation_ap_std":
                        importance_ap.importances_std[
                            feature_index
                        ],

                    "permutation_roc_mean":
                        importance_roc.importances_mean[
                            feature_index
                        ],

                    "permutation_roc_std":
                        importance_roc.importances_std[
                            feature_index
                        ],

                    "impurity_importance":
                        impurity_importance[
                            feature_index
                        ],
                }
            )

    fold_importance = pd.DataFrame(
        fold_rows
    )

    fold_performance = pd.DataFrame(
        fold_performance_rows
    )

    summary = (
        fold_importance
        .groupby(
            [
                "feature",
                "feature_family",
            ]
        )
        .agg(
            permutation_ap_mean=(
                "permutation_ap_mean",
                "mean",
            ),
            permutation_ap_std_across_folds=(
                "permutation_ap_mean",
                "std",
            ),
            permutation_roc_mean=(
                "permutation_roc_mean",
                "mean",
            ),
            permutation_roc_std_across_folds=(
                "permutation_roc_mean",
                "std",
            ),
            impurity_importance_mean=(
                "impurity_importance",
                "mean",
            ),
            positive_ap_fold_count=(
                "permutation_ap_mean",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            positive_roc_fold_count=(
                "permutation_roc_mean",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
        )
        .reset_index()
    )

    summary[
        "mean_rank_ap"
    ] = (
        summary[
            "permutation_ap_mean"
        ]
        .rank(
            ascending=False,
            method="average",
        )
    )

    summary[
        "mean_rank_roc"
    ] = (
        summary[
            "permutation_roc_mean"
        ]
        .rank(
            ascending=False,
            method="average",
        )
    )

    summary[
        "combined_rank"
    ] = (
        summary["mean_rank_ap"]
        + summary["mean_rank_roc"]
    ) / 2

    summary = summary.sort_values(
        [
            "combined_rank",
            "permutation_ap_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )

    family_summary = (
        summary
        .groupby("feature_family")
        .agg(
            feature_count=(
                "feature",
                "count",
            ),
            total_permutation_ap=(
                "permutation_ap_mean",
                "sum",
            ),
            mean_permutation_ap=(
                "permutation_ap_mean",
                "mean",
            ),
            total_permutation_roc=(
                "permutation_roc_mean",
                "sum",
            ),
            mean_permutation_roc=(
                "permutation_roc_mean",
                "mean",
            ),
            total_impurity_importance=(
                "impurity_importance_mean",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "total_permutation_ap",
                "total_permutation_roc",
            ],
            ascending=False,
        )
    )

    FOLD_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_importance.to_csv(
        FOLD_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    family_summary.to_csv(
        FAMILY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RF PERMUTATION IMPORTANCE V1 ===\n"
    )

    print("Rows:", len(df))
    print("Features:", len(model_features))
    print("CV folds:", N_SPLITS)
    print("Permutation repeats:", N_REPEATS)

    print(
        "\n=== OUTER-FOLD MODEL PERFORMANCE ===\n"
    )

    print(
        fold_performance
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nMean ROC-AUC:",
        round(
            fold_performance[
                "roc_auc"
            ].mean(),
            4,
        ),
    )

    print(
        "Mean Average Precision:",
        round(
            fold_performance[
                "average_precision"
            ].mean(),
            4,
        ),
    )

    print(
        "\n=== FEATURE IMPORTANCE SUMMARY ===\n"
    )

    display_columns = [
        "feature",
        "feature_family",
        "permutation_ap_mean",
        "permutation_ap_std_across_folds",
        "positive_ap_fold_count",
        "permutation_roc_mean",
        "permutation_roc_std_across_folds",
        "positive_roc_fold_count",
        "impurity_importance_mean",
        "combined_rank",
    ]

    print(
        summary[
            display_columns
        ]
        .round(5)
        .to_string(index=False)
    )

    print(
        "\n=== FEATURE FAMILY SUMMARY ===\n"
    )

    print(
        family_summary
        .round(5)
        .to_string(index=False)
    )

    print(
        "\n=== FEATURES POSITIVE IN AT LEAST "
        "3 OF 5 FOLDS FOR BOTH METRICS ===\n"
    )

    stable_features = summary[
        (
            summary[
                "positive_ap_fold_count"
            ] >= 3
        )
        & (
            summary[
                "positive_roc_fold_count"
            ] >= 3
        )
    ]

    if stable_features.empty:
        print("None")
    else:
        print(
            stable_features[
                display_columns
            ]
            .round(5)
            .to_string(index=False)
        )

    print(
        "\nImportant interpretation:"
    )

    print(
        "Permutation importance was calculated "
        "only on held-out test folds."
    )

    print(
        "Negative importance means shuffling "
        "occasionally improved held-out performance."
    )

    print(
        "Features with unstable or negative importance "
        "must not be converted into strong recommendations."
    )

    print("\nSaved fold importance:")
    print(FOLD_OUTPUT_FILE)

    print("\nSaved feature summary:")
    print(SUMMARY_OUTPUT_FILE)

    print("\nSaved family summary:")
    print(FAMILY_OUTPUT_FILE)


if __name__ == "__main__":
    main()
