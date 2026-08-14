from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_feature_matrix_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_adjusted_strategy_signal_audit_v1.csv"
)

RESIDUAL_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_performance_residuals_v1.csv"
)


TARGET = "performance_percentile_final"

BINARY_SIGNALS = [
    "title_has_music_video",
    "title_has_lyric_video",
]

DURATION_FEATURE = "duration_minutes"


def analyze_binary(
    df,
    target,
    signal,
    adjustment_type,
):
    zero = df[df[signal] == 0][target]
    one = df[df[signal] == 1][target]

    if len(zero) > 0 and len(one) > 0:
        test = mannwhitneyu(
            one,
            zero,
            alternative="two-sided",
        )

        statistic = float(test.statistic)
        pvalue = float(test.pvalue)
    else:
        statistic = np.nan
        pvalue = np.nan

    return {
        "analysis_type":
            "binary_signal",

        "adjustment_type":
            adjustment_type,

        "signal":
            signal,

        "target":
            target,

        "n_signal_0":
            len(zero),

        "n_signal_1":
            len(one),

        "median_signal_0":
            (
                float(zero.median())
                if len(zero) > 0
                else np.nan
            ),

        "median_signal_1":
            (
                float(one.median())
                if len(one) > 0
                else np.nan
            ),

        "median_difference":
            (
                float(one.median() - zero.median())
                if (
                    len(zero) > 0
                    and len(one) > 0
                )
                else np.nan
            ),

        "mann_whitney_statistic":
            statistic,

        "pvalue":
            pvalue,

        "spearman_correlation":
            np.nan,
    }


def analyze_duration(
    df,
    target,
    adjustment_type,
):
    correlation, pvalue = spearmanr(
        df[DURATION_FEATURE],
        df[target],
    )

    return {
        "analysis_type":
            "continuous_signal",

        "adjustment_type":
            adjustment_type,

        "signal":
            DURATION_FEATURE,

        "target":
            target,

        "n_signal_0":
            np.nan,

        "n_signal_1":
            len(df),

        "median_signal_0":
            np.nan,

        "median_signal_1":
            np.nan,

        "median_difference":
            np.nan,

        "mann_whitney_statistic":
            np.nan,

        "pvalue":
            float(pvalue),

        "spearman_correlation":
            float(correlation),
    }


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

    artist_median = (
        df.groupby("artist_name_fa")[TARGET]
        .transform("median")
    )

    category_median = (
        df.groupby("category")[TARGET]
        .transform("median")
    )

    df[
        "performance_residual_artist_median"
    ] = (
        df[TARGET] - artist_median
    )

    df[
        "performance_residual_category_median"
    ] = (
        df[TARGET] - category_median
    )

    df[
        "artist_performance_median"
    ] = artist_median

    df[
        "category_performance_median"
    ] = category_median

    analyses = [
        (
            "unadjusted",
            TARGET,
        ),
        (
            "artist_median_adjusted",
            "performance_residual_artist_median",
        ),
        (
            "category_median_adjusted",
            "performance_residual_category_median",
        ),
    ]

    rows = []

    for adjustment_type, target in analyses:
        for signal in BINARY_SIGNALS:
            rows.append(
                analyze_binary(
                    df=df,
                    target=target,
                    signal=signal,
                    adjustment_type=adjustment_type,
                )
            )

        rows.append(
            analyze_duration(
                df=df,
                target=target,
                adjustment_type=adjustment_type,
            )
        )

    results = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    residual_columns = [
        "record_id",
        "artist_name_fa",
        "category",
        "api_video_title",
        TARGET,
        "artist_performance_median",
        "category_performance_median",
        "performance_residual_artist_median",
        "performance_residual_category_median",
        "title_has_music_video",
        "title_has_lyric_video",
        "duration_minutes",
    ]

    df[residual_columns].to_csv(
        RESIDUAL_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== ARTIST-ADJUSTED STRATEGY SIGNAL AUDIT V1 ===\n"
    )

    print(
        results
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== BINARY SIGNAL DIRECTION COMPARISON ===\n"
    )

    binary_results = results[
        results["analysis_type"]
        == "binary_signal"
    ]

    print(
        binary_results[
            [
                "adjustment_type",
                "signal",
                "n_signal_0",
                "n_signal_1",
                "median_signal_0",
                "median_signal_1",
                "median_difference",
                "pvalue",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== DURATION DIRECTION COMPARISON ===\n"
    )

    duration_results = results[
        results["analysis_type"]
        == "continuous_signal"
    ]

    print(
        duration_results[
            [
                "adjustment_type",
                "signal",
                "spearman_correlation",
                "pvalue",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== RESIDUAL CENTER CHECK ===\n"
    )

    artist_center_check = (
        df.groupby("artist_name_fa")[
            "performance_residual_artist_median"
        ]
        .median()
        .round(8)
    )

    category_center_check = (
        df.groupby("category")[
            "performance_residual_category_median"
        ]
        .median()
        .round(8)
    )

    print(
        "\nArtist residual medians:"
    )

    print(
        artist_center_check.to_string()
    )

    print(
        "\nCategory residual medians:"
    )

    print(
        category_center_check.to_string()
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "Artist-median adjustment removes "
        "between-artist baseline performance differences."
    )

    print(
        "Category-median adjustment removes "
        "between-category baseline performance differences."
    )

    print(
        "Persistence of signal direction after adjustment "
        "supports a more robust descriptive association."
    )

    print(
        "This remains observational and does not establish causality."
    )

    print("\nSaved adjusted signal audit:")
    print(OUTPUT_FILE)

    print("\nSaved performance residuals:")
    print(RESIDUAL_OUTPUT_FILE)


if __name__ == "__main__":
    main()
