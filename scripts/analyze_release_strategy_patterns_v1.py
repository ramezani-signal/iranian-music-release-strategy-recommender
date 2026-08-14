from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v2_format.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_pattern_analysis_v1.csv"
)


BINARY_STRATEGY_FEATURES = [
    "is_iran_weekend_release",
    "title_has_official",
    "title_has_music_video",
    "title_has_lyric_video",
    "title_has_visual",
    "title_has_clip",
    "title_has_soundtrack",
    "title_has_alternative_version",
    "title_has_collaboration",
    "title_contains_persian",
    "title_contains_latin",
    "title_is_bilingual",
    "title_has_separator",
]


CONTINUOUS_STRATEGY_FEATURES = [
    "release_hour_tehran",
    "days_since_previous_release",
    "title_character_count",
    "title_word_count",
    "duration_minutes",
]


def safe_spearman(df, feature, target):
    subset = (
        df[
            [
                feature,
                target,
            ]
        ]
        .dropna()
    )

    if len(subset) < 3:
        return np.nan

    if subset[feature].nunique() < 2:
        return np.nan

    if subset[target].nunique() < 2:
        return np.nan

    return subset[feature].corr(
        subset[target],
        method="spearman",
    )


def build_binary_summary(df):
    rows = []

    for feature in BINARY_STRATEGY_FEATURES:
        for value in [0, 1]:
            subset = df[
                df[feature] == value
            ]

            if len(subset) == 0:
                continue

            rows.append(
                {
                    "analysis_type":
                        "binary_feature_summary",
                    "feature": feature,
                    "feature_value": value,
                    "group_name": "global",
                    "group_value": "all",
                    "count": len(subset),
                    "median_performance":
                        subset[
                            "performance_percentile_final"
                        ].median(),
                    "mean_performance":
                        subset[
                            "performance_percentile_final"
                        ].mean(),
                    "high_or_top_rate":
                        subset[
                            "is_high_or_top"
                        ].mean(),
                    "spearman_correlation": np.nan,
                }
            )

    return pd.DataFrame(rows)


def build_continuous_summary(df):
    rows = []

    for feature in CONTINUOUS_STRATEGY_FEATURES:
        rows.append(
            {
                "analysis_type":
                    "continuous_feature_correlation",
                "feature": feature,
                "feature_value": np.nan,
                "group_name": "global",
                "group_value": "all",
                "count":
                    df[feature].notna().sum(),
                "median_performance": np.nan,
                "mean_performance": np.nan,
                "high_or_top_rate": np.nan,
                "spearman_correlation":
                    safe_spearman(
                        df,
                        feature,
                        "performance_percentile_final",
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_grouped_binary_summary(
    df,
    group_column,
):
    rows = []

    for group_value, group_df in df.groupby(
        group_column
    ):
        for feature in BINARY_STRATEGY_FEATURES:
            if group_df[feature].nunique() < 2:
                continue

            for value in [0, 1]:
                subset = group_df[
                    group_df[feature] == value
                ]

                if len(subset) == 0:
                    continue

                rows.append(
                    {
                        "analysis_type":
                            "grouped_binary_summary",
                        "feature": feature,
                        "feature_value": value,
                        "group_name": group_column,
                        "group_value": group_value,
                        "count": len(subset),
                        "median_performance":
                            subset[
                                "performance_percentile_final"
                            ].median(),
                        "mean_performance":
                            subset[
                                "performance_percentile_final"
                            ].mean(),
                        "high_or_top_rate":
                            subset[
                                "is_high_or_top"
                            ].mean(),
                        "spearman_correlation":
                            np.nan,
                    }
                )

    return pd.DataFrame(rows)


def build_grouped_continuous_summary(
    df,
    group_column,
):
    rows = []

    for group_value, group_df in df.groupby(
        group_column
    ):
        for feature in CONTINUOUS_STRATEGY_FEATURES:
            correlation = safe_spearman(
                group_df,
                feature,
                "performance_percentile_final",
            )

            rows.append(
                {
                    "analysis_type":
                        "grouped_continuous_correlation",
                    "feature": feature,
                    "feature_value": np.nan,
                    "group_name": group_column,
                    "group_value": group_value,
                    "count":
                        group_df[feature]
                        .notna()
                        .sum(),
                    "median_performance": np.nan,
                    "mean_performance": np.nan,
                    "high_or_top_rate": np.nan,
                    "spearman_correlation":
                        correlation,
                }
            )

    return pd.DataFrame(rows)


def print_binary_differences(
    analysis,
    group_name,
):
    subset = analysis[
        (
            analysis["analysis_type"]
            .isin(
                [
                    "binary_feature_summary",
                    "grouped_binary_summary",
                ]
            )
        )
        & (
            analysis["group_name"]
            == group_name
        )
    ]

    if subset.empty:
        return

    pivot = subset.pivot_table(
        index=[
            "group_value",
            "feature",
        ],
        columns="feature_value",
        values=[
            "count",
            "median_performance",
            "high_or_top_rate",
        ],
        aggfunc="first",
    )

    required_columns = [
        ("median_performance", 0),
        ("median_performance", 1),
    ]

    if not all(
        column in pivot.columns
        for column in required_columns
    ):
        return

    pivot["median_difference_1_minus_0"] = (
        pivot[
            ("median_performance", 1)
        ]
        - pivot[
            ("median_performance", 0)
        ]
    )

    rate_columns = [
        ("high_or_top_rate", 0),
        ("high_or_top_rate", 1),
    ]

    if all(
        column in pivot.columns
        for column in rate_columns
    ):
        pivot["high_rate_difference_1_minus_0"] = (
            (
                pivot[
                    ("high_or_top_rate", 1)
                ]
                - pivot[
                    ("high_or_top_rate", 0)
                ]
            )
            * 100
        )

    print(
        pivot
        .sort_values(
            "median_difference_1_minus_0",
            ascending=False,
        )
        .round(2)
        .to_string()
    )


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
        BINARY_STRATEGY_FEATURES
        + CONTINUOUS_STRATEGY_FEATURES
        + [
            "artist_name_fa",
            "category",
            "source_type",
            "performance_percentile_final",
            "is_high_or_top",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    analysis_frames = [
        build_binary_summary(df),
        build_continuous_summary(df),
        build_grouped_binary_summary(
            df,
            "category",
        ),
        build_grouped_binary_summary(
            df,
            "artist_name_fa",
        ),
        build_grouped_continuous_summary(
            df,
            "category",
        ),
        build_grouped_continuous_summary(
            df,
            "artist_name_fa",
        ),
    ]

    analysis = pd.concat(
        analysis_frames,
        ignore_index=True,
    )

    analysis["high_or_top_rate_percent"] = (
        analysis["high_or_top_rate"]
        * 100
    )

    analysis["strategy_analysis_version"] = (
        "STRATEGY_PATTERN_V1"
    )

    analysis.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RELEASE STRATEGY PATTERN ANALYSIS V1 ===\n"
    )

    print("Input rows:", len(df))
    print(
        "Analysis rows:",
        len(analysis),
    )

    print(
        "\n=== GLOBAL BINARY FEATURE DIFFERENCES ===\n"
    )

    print_binary_differences(
        analysis,
        "global",
    )

    print(
        "\n=== CATEGORY-CONTROLLED BINARY DIFFERENCES ===\n"
    )

    print_binary_differences(
        analysis,
        "category",
    )

    print(
        "\n=== ARTIST-CONTROLLED BINARY DIFFERENCES ===\n"
    )

    print_binary_differences(
        analysis,
        "artist_name_fa",
    )

    print(
        "\n=== GLOBAL CONTINUOUS SPEARMAN CORRELATIONS ===\n"
    )

    global_corr = analysis[
        analysis["analysis_type"]
        == "continuous_feature_correlation"
    ][
        [
            "feature",
            "count",
            "spearman_correlation",
        ]
    ]

    print(
        global_corr
        .sort_values(
            "spearman_correlation",
            ascending=False,
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== CATEGORY-CONTROLLED CONTINUOUS CORRELATIONS ===\n"
    )

    category_corr = analysis[
        (
            analysis["analysis_type"]
            == "grouped_continuous_correlation"
        )
        & (
            analysis["group_name"]
            == "category"
        )
    ][
        [
            "group_value",
            "feature",
            "count",
            "spearman_correlation",
        ]
    ]

    print(
        category_corr
        .sort_values(
            [
                "group_value",
                "spearman_correlation",
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
        "\n=== ARTIST-CONTROLLED CONTINUOUS CORRELATIONS ===\n"
    )

    artist_corr = analysis[
        (
            analysis["analysis_type"]
            == "grouped_continuous_correlation"
        )
        & (
            analysis["group_name"]
            == "artist_name_fa"
        )
    ][
        [
            "group_value",
            "feature",
            "count",
            "spearman_correlation",
        ]
    ]

    print(
        artist_corr
        .sort_values(
            [
                "group_value",
                "spearman_correlation",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
