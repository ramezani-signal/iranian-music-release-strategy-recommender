from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v2_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "performance_components_audit_v1.csv"
)


COMPONENTS = [
    "log_views_per_day",
    "engagement_rate",
    "likes_per_1000_views",
    "comments_per_1000_views",
]


def robust_zscore(series):
    median = series.median()

    mad = np.median(
        np.abs(series - median)
    )

    if mad == 0 or np.isnan(mad):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
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

    print(
        "\n=== CATEGORY SAMPLE SIZES ===\n"
    )

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    for component in COMPONENTS:
        output_col = (
            f"category_robust_z_{component}"
        )

        df[output_col] = (
            df.groupby(
                "category",
                group_keys=False,
            )[component]
            .transform(robust_zscore)
        )

    ROBUST_COMPONENTS = [
        f"category_robust_z_{component}"
        for component in COMPONENTS
    ]

    if df[ROBUST_COMPONENTS].isna().sum().sum() != 0:
        raise ValueError(
            "Missing robust component values."
        )

    if not np.isfinite(
        df[ROBUST_COMPONENTS]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite robust component values."
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RAW COMPONENT CORRELATION ===\n"
    )

    print(
        df[COMPONENTS]
        .corr(method="spearman")
        .round(3)
        .to_string()
    )

    print(
        "\n=== CATEGORY-ROBUST COMPONENT "
        "CORRELATION ===\n"
    )

    print(
        df[ROBUST_COMPONENTS]
        .corr(method="spearman")
        .round(3)
        .to_string()
    )

    print(
        "\n=== ROBUST COMPONENT SUMMARY ===\n"
    )

    print(
        df[ROBUST_COMPONENTS]
        .describe()
        .transpose()
        .to_string()
    )

    print(
        "\n=== ROBUST COMPONENT MEDIANS "
        "BY CATEGORY ===\n"
    )

    print(
        df.groupby("category")[
            ROBUST_COMPONENTS
        ]
        .median()
        .round(6)
        .to_string()
    )

    print(
        "\n=== TOP 20 ABSOLUTE ROBUST "
        "OUTLIERS ===\n"
    )

    df["max_abs_robust_component"] = (
        df[ROBUST_COMPONENTS]
        .abs()
        .max(axis=1)
    )

    print(
        df.sort_values(
            "max_abs_robust_component",
            ascending=False,
        )[
            [
                "artist_name_fa",
                "category",
                "api_video_title",
                "view_count",
                "views_per_day",
                "engagement_rate",
                *ROBUST_COMPONENTS,
                "max_abs_robust_component",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        "\n=== COMPONENT QUANTILES ===\n"
    )

    quantiles = df[
        ROBUST_COMPONENTS
    ].quantile(
        [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print(
        quantiles.round(3).to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
