from pathlib import Path
import json

import pandas as pd


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_evaluation_protocol_audit_v1.csv"
)


TARGET = "is_high_or_top"
SECONDARY_TARGET = "is_top_performer"
GROUP_COLUMN = "artist_name_fa"


def main():
    try:
        import sklearn
        from sklearn.model_selection import (
            LeaveOneGroupOut,
            StratifiedKFold,
        )
    except ImportError as exc:
        raise SystemExit(
            "ERROR: scikit-learn is not installed.\n"
            "Run: pip install scikit-learn"
        ) from exc

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

    required_columns = (
        [
            "record_id",
            "artist_name_fa",
            "category",
            TARGET,
            SECONDARY_TARGET,
        ]
        + model_features
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    print(
        "\n=== MODEL EVALUATION PROTOCOL AUDIT V1 ===\n"
    )

    print("scikit-learn version:")
    print(sklearn.__version__)

    print("\nRows:", len(df))
    print("Model feature count:", len(model_features))
    print("Artist groups:", df[GROUP_COLUMN].nunique())

    print("\nPrimary target counts:\n")
    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPrimary target prevalence:")
    print(
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    print("\nSecondary target counts:\n")
    print(
        df[SECONDARY_TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    artist_summary = (
        df.groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )
        .agg(
            rows=(
                "record_id",
                "count",
            ),
            high_or_top_positive=(
                TARGET,
                "sum",
            ),
            top_performer_positive=(
                SECONDARY_TARGET,
                "sum",
            ),
        )
        .reset_index()
    )

    artist_summary[
        "high_or_top_negative"
    ] = (
        artist_summary["rows"]
        - artist_summary[
            "high_or_top_positive"
        ]
    )

    artist_summary[
        "high_or_top_rate_percent"
    ] = (
        artist_summary[
            "high_or_top_positive"
        ]
        / artist_summary["rows"]
        * 100
    ).round(2)

    artist_summary[
        "test_fold_has_both_classes"
    ] = (
        (
            artist_summary[
                "high_or_top_positive"
            ] > 0
        )
        & (
            artist_summary[
                "high_or_top_negative"
            ] > 0
        )
    ).astype(int)

    artist_summary = artist_summary.sort_values(
        [
            "category",
            "artist_name_fa",
        ]
    )

    print(
        "\n=== PRIMARY TARGET BY ARTIST ===\n"
    )

    print(
        artist_summary.to_string(
            index=False
        )
    )

    print(
        "\nArtists whose held-out fold "
        "contains both target classes:"
    )

    print(
        artist_summary[
            "test_fold_has_both_classes"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    category_summary = (
        df.groupby("category")
        .agg(
            rows=(
                "record_id",
                "count",
            ),
            high_or_top_positive=(
                TARGET,
                "sum",
            ),
        )
        .reset_index()
    )

    category_summary[
        "high_or_top_negative"
    ] = (
        category_summary["rows"]
        - category_summary[
            "high_or_top_positive"
        ]
    )

    category_summary[
        "high_or_top_rate_percent"
    ] = (
        category_summary[
            "high_or_top_positive"
        ]
        / category_summary["rows"]
        * 100
    ).round(2)

    print(
        "\n=== PRIMARY TARGET BY CATEGORY ===\n"
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    feature_duplicate_rows = (
        df.duplicated(
            subset=model_features,
            keep=False,
        )
        .sum()
    )

    print(
        "\nRows participating in duplicate "
        "model-feature vectors:"
    )

    print(feature_duplicate_rows)

    logo = LeaveOneGroupOut()

    logo_rows = []

    X = df[model_features]
    y = df[TARGET]
    groups = df[GROUP_COLUMN]

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        logo.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        held_out_artist = (
            test[GROUP_COLUMN].iloc[0]
        )

        logo_rows.append(
            {
                "protocol":
                    "leave_one_artist_out",
                "fold":
                    fold_number,
                "held_out_artist":
                    held_out_artist,
                "train_rows":
                    len(train),
                "test_rows":
                    len(test),
                "train_positive":
                    int(train[TARGET].sum()),
                "train_negative":
                    int(
                        len(train)
                        - train[TARGET].sum()
                    ),
                "test_positive":
                    int(test[TARGET].sum()),
                "test_negative":
                    int(
                        len(test)
                        - test[TARGET].sum()
                    ),
                "test_has_both_classes":
                    int(
                        test[TARGET].nunique()
                        == 2
                    ),
            }
        )

    logo_audit = pd.DataFrame(
        logo_rows
    )

    print(
        "\n=== LEAVE-ONE-ARTIST-OUT FOLDS ===\n"
    )

    print(
        logo_audit.to_string(
            index=False
        )
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    stratified_rows = []

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        artist_overlap = (
            set(train[GROUP_COLUMN])
            & set(test[GROUP_COLUMN])
        )

        stratified_rows.append(
            {
                "protocol":
                    "stratified_5_fold",
                "fold":
                    fold_number,
                "held_out_artist":
                    "",
                "train_rows":
                    len(train),
                "test_rows":
                    len(test),
                "train_positive":
                    int(train[TARGET].sum()),
                "train_negative":
                    int(
                        len(train)
                        - train[TARGET].sum()
                    ),
                "test_positive":
                    int(test[TARGET].sum()),
                "test_negative":
                    int(
                        len(test)
                        - test[TARGET].sum()
                    ),
                "test_has_both_classes":
                    int(
                        test[TARGET].nunique()
                        == 2
                    ),
                "artist_overlap_count":
                    len(artist_overlap),
            }
        )

    stratified_audit = pd.DataFrame(
        stratified_rows
    )

    print(
        "\n=== STRATIFIED 5-FOLD AUDIT ===\n"
    )

    print(
        stratified_audit.to_string(
            index=False
        )
    )

    protocol_audit = pd.concat(
        [
            logo_audit,
            stratified_audit,
        ],
        ignore_index=True,
    )

    protocol_audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nRecommended interpretation:"
    )

    print(
        "1. Stratified 5-fold estimates "
        "record-level prediction for known artist populations."
    )

    print(
        "2. Leave-one-artist-out estimates "
        "transfer to an unseen artist."
    )

    print(
        "3. ROC-AUC/PR-AUC cannot be computed "
        "for held-out folds containing only one class."
    )

    print(
        "4. Balanced accuracy, recall, "
        "specificity and confusion counts must also be reported."
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
