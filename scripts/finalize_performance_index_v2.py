from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "performance_index_sensitivity_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v4_final_performance.csv"
)


VIEW_COMPONENT = "pi_view_component_clipped"
ENGAGEMENT_COMPONENT = "pi_engagement_component_clipped"
COMMENT_COMPONENT = "pi_comment_component_clipped"


WEIGHT_VIEW = 0.65
WEIGHT_ENGAGEMENT = 0.25
WEIGHT_COMMENT = 0.10


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    required_columns = [
        "record_id",
        "artist_id",
        "artist_name_fa",
        "category",
        "video_id",
        "api_video_title",
        VIEW_COMPONENT,
        ENGAGEMENT_COMPONENT,
        COMMENT_COMPONENT,
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["performance_index_raw_final"] = (
        WEIGHT_VIEW
        * df[VIEW_COMPONENT]
        + WEIGHT_ENGAGEMENT
        * df[ENGAGEMENT_COMPONENT]
        + WEIGHT_COMMENT
        * df[COMMENT_COMPONENT]
    )

    df["performance_percentile_final"] = (
        df.groupby("category")[
            "performance_index_raw_final"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    ).round(2)

    df["performance_tier_final"] = pd.cut(
        df["performance_percentile_final"],
        bins=[
            0,
            25,
            50,
            75,
            90,
            100,
        ],
        labels=[
            "low",
            "below_median",
            "above_median",
            "high",
            "top_performer",
        ],
        include_lowest=True,
    )

    df["performance_index_version"] = (
        "PI_V2_FINAL_65_25_10"
    )

    df["performance_index_view_weight"] = (
        WEIGHT_VIEW
    )

    df[
        "performance_index_engagement_weight"
    ] = WEIGHT_ENGAGEMENT

    df["performance_index_comment_weight"] = (
        WEIGHT_COMMENT
    )

    df["performance_index_clip_lower"] = -3.0
    df["performance_index_clip_upper"] = 3.0

    check_columns = [
        "performance_index_raw_final",
        "performance_percentile_final",
    ]

    if df[check_columns].isna().sum().sum() != 0:
        raise ValueError(
            "Missing values found in final PI."
        )

    if not np.isfinite(
        df[check_columns]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in final PI."
        )

    if not df[
        "performance_percentile_final"
    ].between(0, 100).all():
        raise ValueError(
            "Final percentile outside 0-100."
        )

    expected_raw = df["pi_raw_C_65_25_10"]

    max_raw_difference = (
        df["performance_index_raw_final"]
        - expected_raw
    ).abs().max()

    if max_raw_difference > 1e-12:
        raise ValueError(
            "Final PI does not match "
            "sensitivity scheme C."
        )

    expected_pct = df["pi_pct_C_65_25_10"]

    max_pct_difference = (
        df["performance_percentile_final"]
        - expected_pct
    ).abs().max()

    if max_pct_difference > 1e-9:
        raise ValueError(
            "Final percentile does not match "
            "sensitivity scheme C."
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== FINAL PERFORMANCE INDEX V2 ===\n"
    )

    print("Rows:", len(df))

    print(
        "\nFinal specification:"
    )

    print(
        "PI = 0.65 * View "
        "+ 0.25 * Engagement "
        "+ 0.10 * Comment"
    )

    print(
        "Component clipping: [-3, +3]"
    )

    print(
        "Ranking: category-relative percentile"
    )

    print(
        "\nVersion:"
    )

    print(
        df["performance_index_version"]
        .unique()
    )

    print(
        "\nMaximum raw difference "
        "from sensitivity scheme C:"
    )

    print(max_raw_difference)

    print(
        "\nMaximum percentile difference "
        "from sensitivity scheme C:"
    )

    print(max_pct_difference)

    print(
        "\nFinal PI summary:\n"
    )

    print(
        df[
            [
                "performance_index_raw_final",
                "performance_percentile_final",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\nFinal tier counts:\n"
    )

    print(
        df["performance_tier_final"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal tiers by category:\n"
    )

    print(
        pd.crosstab(
            df["category"],
            df["performance_tier_final"],
        ).to_string()
    )

    print(
        "\n=== TOP 20 FINAL PERFORMANCE ===\n"
    )

    print(
        df.sort_values(
            [
                "performance_percentile_final",
                "performance_index_raw_final",
            ],
            ascending=False,
        )[
            [
                "artist_name_fa",
                "category",
                "api_video_title",
                "views_per_day",
                "engagement_rate",
                "comments_per_1000_views",
                "performance_index_raw_final",
                "performance_percentile_final",
                "performance_tier_final",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\n=== FINAL ARTIST PERFORMANCE SUMMARY ===\n"
    )

    artist_summary = (
        df.groupby(
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
            median_performance_percentile=(
                "performance_percentile_final",
                "median",
            ),
            mean_performance_percentile=(
                "performance_percentile_final",
                "mean",
            ),
            top_performer_count=(
                "performance_tier_final",
                lambda x: (
                    x == "top_performer"
                ).sum(),
            ),
            high_or_top_count=(
                "performance_tier_final",
                lambda x: x.isin(
                    [
                        "high",
                        "top_performer",
                    ]
                ).sum(),
            ),
        )
        .reset_index()
        .sort_values(
            "median_performance_percentile",
            ascending=False,
        )
    )

    print(
        artist_summary.to_string(index=False)
    )

    print(
        "\n=== FINAL COMPONENT TO INDEX "
        "SPEARMAN CORRELATION ===\n"
    )

    correlation_columns = [
        "log_views_per_day",
        "engagement_rate",
        "comments_per_1000_views",
        "performance_index_raw_final",
        "performance_percentile_final",
    ]

    print(
        df[correlation_columns]
        .corr(method="spearman")
        .round(4)
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
