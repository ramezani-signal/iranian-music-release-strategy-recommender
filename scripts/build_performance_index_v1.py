from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "performance_components_audit_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v3_performance_index.csv"
)


VIEW_COMPONENT = (
    "category_robust_z_log_views_per_day"
)

ENGAGEMENT_COMPONENT = (
    "category_robust_z_engagement_rate"
)

COMMENT_COMPONENT = (
    "category_robust_z_comments_per_1000_views"
)


CLIP_LOWER = -3.0
CLIP_UPPER = 3.0


WEIGHT_VIEW = 0.60
WEIGHT_ENGAGEMENT = 0.25
WEIGHT_COMMENT = 0.15


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    required_columns = [
        "record_id",
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
            "Missing required columns: "
            f"{missing_columns}"
        )

    df["pi_view_component_clipped"] = (
        df[VIEW_COMPONENT]
        .clip(
            lower=CLIP_LOWER,
            upper=CLIP_UPPER,
        )
    )

    df["pi_engagement_component_clipped"] = (
        df[ENGAGEMENT_COMPONENT]
        .clip(
            lower=CLIP_LOWER,
            upper=CLIP_UPPER,
        )
    )

    df["pi_comment_component_clipped"] = (
        df[COMMENT_COMPONENT]
        .clip(
            lower=CLIP_LOWER,
            upper=CLIP_UPPER,
        )
    )

    df["performance_index_raw"] = (
        WEIGHT_VIEW
        * df["pi_view_component_clipped"]
        + WEIGHT_ENGAGEMENT
        * df["pi_engagement_component_clipped"]
        + WEIGHT_COMMENT
        * df["pi_comment_component_clipped"]
    )

    df["performance_percentile_category"] = (
        df.groupby("category")[
            "performance_index_raw"
        ]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    df["performance_percentile_category"] = (
        df["performance_percentile_category"]
        .round(2)
    )

    df["performance_tier"] = pd.cut(
        df["performance_percentile_category"],
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

    check_columns = [
        "pi_view_component_clipped",
        "pi_engagement_component_clipped",
        "pi_comment_component_clipped",
        "performance_index_raw",
        "performance_percentile_category",
    ]

    if df[check_columns].isna().sum().sum() != 0:
        raise ValueError(
            "Missing values found in performance index."
        )

    if not np.isfinite(
        df[check_columns]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite values found in performance index."
        )

    if not df[
        "performance_percentile_category"
    ].between(0, 100).all():
        raise ValueError(
            "Performance percentile outside 0-100."
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== PERFORMANCE INDEX V1 ===\n"
    )

    print("Rows:", len(df))

    print(
        "\nWeights:"
    )

    print(
        f"View velocity: {WEIGHT_VIEW:.2f}"
    )

    print(
        f"Engagement rate: "
        f"{WEIGHT_ENGAGEMENT:.2f}"
    )

    print(
        f"Comment intensity: "
        f"{WEIGHT_COMMENT:.2f}"
    )

    print(
        "\nClipping range:"
    )

    print(
        CLIP_LOWER,
        "to",
        CLIP_UPPER,
    )

    print(
        "\nPerformance index summary:\n"
    )

    print(
        df[
            [
                "performance_index_raw",
                "performance_percentile_category",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\nPerformance tier counts:\n"
    )

    print(
        df["performance_tier"]
        .value_counts()
        .to_string()
    )

    print(
        "\nPerformance tiers by category:\n"
    )

    print(
        pd.crosstab(
            df["category"],
            df["performance_tier"],
        ).to_string()
    )

    print(
        "\nMedian percentile by category:\n"
    )

    print(
        df.groupby("category")[
            "performance_percentile_category"
        ]
        .median()
        .to_string()
    )

    print(
        "\n=== TOP 20 PERFORMANCE RECORDS ===\n"
    )

    print(
        df.sort_values(
            [
                "performance_percentile_category",
                "performance_index_raw",
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
                "performance_index_raw",
                "performance_percentile_category",
                "performance_tier",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\n=== PERFORMANCE SUMMARY BY ARTIST ===\n"
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
                "performance_percentile_category",
                "median",
            ),
            mean_performance_percentile=(
                "performance_percentile_category",
                "mean",
            ),
            top_performer_count=(
                "performance_tier",
                lambda x: (
                    x == "top_performer"
                ).sum(),
            ),
            high_or_top_count=(
                "performance_tier",
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
        "\n=== COMPONENT TO INDEX "
        "SPEARMAN CORRELATION ===\n"
    )

    correlation_columns = [
        "log_views_per_day",
        "engagement_rate",
        "comments_per_1000_views",
        "performance_index_raw",
        "performance_percentile_category",
    ]

    print(
        df[correlation_columns]
        .corr(method="spearman")
        .round(3)
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
