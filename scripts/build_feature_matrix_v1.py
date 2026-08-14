from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v4_final_performance.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_feature_matrix_v1.csv"
)


NUMERIC_FEATURES = [
    "video_age_years",
    "duration_minutes",
    "log_view_count",
    "log_like_count",
    "log_comment_count",
    "log_views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement_rate",
    "performance_index_raw_final",
    "performance_percentile_final",
]


MODEL_BASE_FEATURES = [
    "video_age_years",
    "duration_minutes",
    "log_views_per_day",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "engagement_rate",
]


def robust_zscore(series):
    median = series.median()

    mad = (
        series - median
    ).abs().median()

    if mad == 0 or pd.isna(mad):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
            dtype=float,
        )

    return (
        0.67448975
        * (series - median)
        / mad
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id found."
        )

    if df["video_id"].duplicated().any():
        raise ValueError(
            "Duplicate video_id found."
        )

    required_columns = [
        "record_id",
        "artist_id",
        "artist_name_fa",
        "category",
        "video_id",
        "api_video_title",
        "api_channel_title",
        "published_at",
        "source_type",
        "source_status",
        "performance_index_version",
    ] + NUMERIC_FEATURES

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if (
        df["performance_index_version"]
        .nunique()
        != 1
    ):
        raise ValueError(
            "Multiple PI versions found."
        )

    expected_pi_version = (
        "PI_V2_FINAL_65_25_10"
    )

    actual_pi_version = (
        df["performance_index_version"]
        .iloc[0]
    )

    if actual_pi_version != expected_pi_version:
        raise ValueError(
            "Unexpected PI version: "
            f"{actual_pi_version}"
        )

    matrix = df.copy()

    for feature in MODEL_BASE_FEATURES:
        matrix[
            f"rz_global_{feature}"
        ] = robust_zscore(
            matrix[feature]
        )

        matrix[
            f"rz_category_{feature}"
        ] = (
            matrix
            .groupby(
                "category",
                group_keys=False,
            )[feature]
            .transform(robust_zscore)
        )

    category_dummies = pd.get_dummies(
        matrix["category"],
        prefix="category",
        dtype=int,
    )

    matrix = pd.concat(
        [
            matrix,
            category_dummies,
        ],
        axis=1,
    )

    matrix[
        "is_top_performer"
    ] = (
        matrix[
            "performance_tier_final"
        ]
        == "top_performer"
    ).astype(int)

    matrix[
        "is_high_or_top"
    ] = (
        matrix[
            "performance_tier_final"
        ]
        .isin(
            [
                "high",
                "top_performer",
            ]
        )
    ).astype(int)

    matrix[
        "feature_matrix_version"
    ] = "FM_V1"

    robust_columns = [
        col
        for col in matrix.columns
        if col.startswith("rz_")
    ]

    dummy_columns = [
        col
        for col in matrix.columns
        if col.startswith("category_")
    ]

    check_columns = (
        NUMERIC_FEATURES
        + robust_columns
        + dummy_columns
        + [
            "is_top_performer",
            "is_high_or_top",
        ]
    )

    missing_count = (
        matrix[check_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            "Missing values found in "
            "feature matrix."
        )

    numeric_array = (
        matrix[check_columns]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise ValueError(
            "Infinite values found in "
            "feature matrix."
        )

    matrix.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RELEASE FEATURE MATRIX V1 ===\n"
    )

    print("Rows:", len(matrix))
    print("Columns:", len(matrix.columns))

    print(
        "\nFeature matrix version:"
    )

    print(
        matrix[
            "feature_matrix_version"
        ]
        .unique()
    )

    print(
        "\nPerformance index version:"
    )

    print(
        matrix[
            "performance_index_version"
        ]
        .unique()
    )

    print(
        "\nBase model features:\n"
    )

    for col in MODEL_BASE_FEATURES:
        print(col)

    print(
        "\nRobust standardized columns:"
    )

    print(len(robust_columns))

    for col in robust_columns:
        print(col)

    print(
        "\nCategory dummy columns:"
    )

    print(dummy_columns)

    print(
        "\nMissing values in model fields:"
    )

    print(
        matrix[check_columns]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nTarget counts:"
    )

    print(
        "\nis_top_performer:"
    )

    print(
        matrix["is_top_performer"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nis_high_or_top:"
    )

    print(
        matrix["is_high_or_top"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nCategory counts:"
    )

    print(
        matrix["category"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== GLOBAL ROBUST FEATURE SUMMARY ===\n"
    )

    global_robust_columns = [
        col
        for col in robust_columns
        if col.startswith("rz_global_")
    ]

    print(
        matrix[global_robust_columns]
        .describe()
        .round(4)
        .to_string()
    )

    print(
        "\n=== CATEGORY ROBUST FEATURE SUMMARY ===\n"
    )

    category_robust_columns = [
        col
        for col in robust_columns
        if col.startswith("rz_category_")
    ]

    print(
        matrix.groupby("category")[
            category_robust_columns
        ]
        .median()
        .round(6)
        .to_string()
    )

    print(
        "\n=== FEATURE CORRELATION WITH "
        "FINAL PERFORMANCE PERCENTILE ===\n"
    )

    correlation_features = (
        MODEL_BASE_FEATURES
        + global_robust_columns
        + category_robust_columns
    )

    correlations = (
        matrix[
            correlation_features
            + [
                "performance_percentile_final"
            ]
        ]
        .corr(method="spearman")[
            "performance_percentile_final"
        ]
        .drop(
            "performance_percentile_final"
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        correlations
        .round(4)
        .to_string()
    )

    print(
        "\n=== ARTIST TARGET SUMMARY ===\n"
    )

    artist_summary = (
        matrix.groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )
        .agg(
            release_count=(
                "video_id",
                "count",
            ),
            median_performance=(
                "performance_percentile_final",
                "median",
            ),
            top_performer_count=(
                "is_top_performer",
                "sum",
            ),
            high_or_top_count=(
                "is_high_or_top",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "median_performance",
            ascending=False,
        )
    )

    print(
        artist_summary
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
