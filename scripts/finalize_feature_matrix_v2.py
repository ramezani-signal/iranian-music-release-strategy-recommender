from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_feature_matrix_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_feature_matrix_v2_final.csv"
)


TRUE_CATEGORY_DUMMIES = [
    "category_classic_fusion",
    "category_pop",
    "category_rap_hiphop",
]

POST_RELEASE_ANALYTIC_FEATURES = [
    "log_views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement_rate",
    "rz_category_log_views_per_day",
    "rz_category_likes_per_1000_views",
    "rz_category_comments_per_1000_views",
    "rz_category_engagement_rate",
]

CURRENT_PRE_RELEASE_SAFE_FEATURES = [
    "duration_minutes",
    "rz_global_duration_minutes",
    "rz_category_duration_minutes",
    *TRUE_CATEGORY_DUMMIES,
]

TARGET_COLUMNS = [
    "performance_index_raw_final",
    "performance_percentile_final",
    "performance_tier_final",
    "is_top_performer",
    "is_high_or_top",
]


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    required_columns = (
        TRUE_CATEGORY_DUMMIES
        + POST_RELEASE_ANALYTIC_FEATURES
        + CURRENT_PRE_RELEASE_SAFE_FEATURES
        + TARGET_COLUMNS
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

    df["feature_matrix_version"] = (
        "FM_V2_FINAL_SPLIT_FEATURE_CONTRACT"
    )

    df["post_release_feature_contract"] = (
        "|".join(POST_RELEASE_ANALYTIC_FEATURES)
    )

    df["pre_release_safe_feature_contract"] = (
        "|".join(CURRENT_PRE_RELEASE_SAFE_FEATURES)
    )

    df["target_feature_contract"] = (
        "|".join(TARGET_COLUMNS)
    )

    all_check_columns = list(
        dict.fromkeys(
            TRUE_CATEGORY_DUMMIES
            + POST_RELEASE_ANALYTIC_FEATURES
            + CURRENT_PRE_RELEASE_SAFE_FEATURES
            + [
                "performance_index_raw_final",
                "performance_percentile_final",
                "is_top_performer",
                "is_high_or_top",
            ]
        )
    )

    if df[all_check_columns].isna().sum().sum() != 0:
        raise ValueError(
            "Missing values found in contracted fields."
        )

    numeric_values = df[
        all_check_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Infinite values found in contracted fields."
        )

    invalid_dummy_rows = (
        df[TRUE_CATEGORY_DUMMIES].sum(axis=1)
        != 1
    )

    if invalid_dummy_rows.any():
        print(
            "\nERROR: Invalid category dummy encoding:\n"
        )

        print(
            df.loc[
                invalid_dummy_rows,
                [
                    "record_id",
                    "artist_name_fa",
                    "category",
                    *TRUE_CATEGORY_DUMMIES,
                ],
            ].to_string(index=False)
        )

        raise ValueError(
            "Each row must have exactly one category dummy."
        )

    leakage_overlap = sorted(
        set(POST_RELEASE_ANALYTIC_FEATURES)
        & set(CURRENT_PRE_RELEASE_SAFE_FEATURES)
    )

    if leakage_overlap:
        raise ValueError(
            "Feature contracts overlap: "
            f"{leakage_overlap}"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RELEASE FEATURE MATRIX V2 FINAL ===\n"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nFeature matrix version:")
    print(
        df["feature_matrix_version"]
        .unique()
    )

    print("\nTrue category dummy columns:")
    for column in TRUE_CATEGORY_DUMMIES:
        print(column)

    print("\nCategory dummy row-sum counts:")
    print(
        df[TRUE_CATEGORY_DUMMIES]
        .sum(axis=1)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPost-release analytic features "
        "(not allowed in pre-release prediction):"
    )

    for column in POST_RELEASE_ANALYTIC_FEATURES:
        print(column)

    print(
        "\nCurrent pre-release-safe features:"
    )

    for column in CURRENT_PRE_RELEASE_SAFE_FEATURES:
        print(column)

    print("\nTarget columns:")
    for column in TARGET_COLUMNS:
        print(column)

    print(
        "\nOverlap between pre-release and "
        "post-release contracts:"
    )
    print(leakage_overlap)

    print(
        "\nMissing values in contracted numeric fields:"
    )

    print(
        df[all_check_columns]
        .isna()
        .sum()
        .to_string()
    )

    print("\nTarget counts:")

    print("\nis_top_performer:")
    print(
        df["is_top_performer"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nis_high_or_top:")
    print(
        df["is_high_or_top"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
