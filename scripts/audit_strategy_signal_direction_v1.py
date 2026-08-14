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

OUTPUT_BINARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_signal_binary_direction_v1.csv"
)

OUTPUT_DURATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_signal_duration_direction_v1.csv"
)

OUTPUT_ARTIST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_signal_artist_direction_v1.csv"
)


TARGET_BINARY = "is_high_or_top"

TARGET_CONTINUOUS = (
    "performance_percentile_final"
)

BINARY_SIGNALS = [
    "title_has_music_video",
    "title_has_lyric_video",
]


def safe_rate(series):
    if len(series) == 0:
        return np.nan

    return float(series.mean())


def analyze_binary_signal(
    df,
    signal,
    group_type,
    group_value,
):
    subset_zero = df[
        df[signal] == 0
    ]

    subset_one = df[
        df[signal] == 1
    ]

    if (
        len(subset_zero) > 0
        and len(subset_one) > 0
    ):
        mann_whitney = mannwhitneyu(
            subset_one[
                TARGET_CONTINUOUS
            ],
            subset_zero[
                TARGET_CONTINUOUS
            ],
            alternative="two-sided",
        )

        mw_statistic = (
            float(mann_whitney.statistic)
        )

        mw_pvalue = (
            float(mann_whitney.pvalue)
        )
    else:
        mw_statistic = np.nan
        mw_pvalue = np.nan

    rate_zero = safe_rate(
        subset_zero[TARGET_BINARY]
    )

    rate_one = safe_rate(
        subset_one[TARGET_BINARY]
    )

    median_zero = (
        subset_zero[
            TARGET_CONTINUOUS
        ].median()
        if len(subset_zero) > 0
        else np.nan
    )

    median_one = (
        subset_one[
            TARGET_CONTINUOUS
        ].median()
        if len(subset_one) > 0
        else np.nan
    )

    return {
        "group_type":
            group_type,

        "group_value":
            group_value,

        "signal":
            signal,

        "n_signal_0":
            len(subset_zero),

        "n_signal_1":
            len(subset_one),

        "high_or_top_rate_signal_0":
            rate_zero,

        "high_or_top_rate_signal_1":
            rate_one,

        "high_or_top_rate_difference":
            (
                rate_one - rate_zero
                if (
                    not np.isnan(rate_zero)
                    and not np.isnan(rate_one)
                )
                else np.nan
            ),

        "median_performance_signal_0":
            median_zero,

        "median_performance_signal_1":
            median_one,

        "median_performance_difference":
            (
                median_one - median_zero
                if (
                    not np.isnan(median_zero)
                    and not np.isnan(median_one)
                )
                else np.nan
            ),

        "mann_whitney_statistic":
            mw_statistic,

        "mann_whitney_pvalue":
            mw_pvalue,
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

    binary_rows = []

    for signal in BINARY_SIGNALS:
        binary_rows.append(
            analyze_binary_signal(
                df=df,
                signal=signal,
                group_type="global",
                group_value="all",
            )
        )

    for category, category_df in df.groupby(
        "category"
    ):
        for signal in BINARY_SIGNALS:
            binary_rows.append(
                analyze_binary_signal(
                    df=category_df,
                    signal=signal,
                    group_type="category",
                    group_value=category,
                )
            )

    artist_rows = []

    for artist, artist_df in df.groupby(
        "artist_name_fa"
    ):
        for signal in BINARY_SIGNALS:
            result = analyze_binary_signal(
                df=artist_df,
                signal=signal,
                group_type="artist",
                group_value=artist,
            )

            artist_rows.append(result)

    binary_results = pd.DataFrame(
        binary_rows
    )

    artist_results = pd.DataFrame(
        artist_rows
    )

    duration_rows = []

    duration_features = [
        "duration_minutes",
        "strategy_rz_duration_minutes",
    ]

    for feature in duration_features:
        correlation, pvalue = spearmanr(
            df[feature],
            df[TARGET_CONTINUOUS],
        )

        duration_rows.append(
            {
                "group_type":
                    "global",

                "group_value":
                    "all",

                "duration_feature":
                    feature,

                "rows":
                    len(df),

                "spearman_correlation":
                    float(correlation),

                "spearman_pvalue":
                    float(pvalue),
            }
        )

    for category, category_df in df.groupby(
        "category"
    ):
        for feature in duration_features:
            correlation, pvalue = spearmanr(
                category_df[feature],
                category_df[
                    TARGET_CONTINUOUS
                ],
            )

            duration_rows.append(
                {
                    "group_type":
                        "category",

                    "group_value":
                        category,

                    "duration_feature":
                        feature,

                    "rows":
                        len(category_df),

                    "spearman_correlation":
                        float(correlation),

                    "spearman_pvalue":
                        float(pvalue),
                }
            )

    for artist, artist_df in df.groupby(
        "artist_name_fa"
    ):
        for feature in duration_features:
            if artist_df[feature].nunique() < 2:
                correlation = np.nan
                pvalue = np.nan
            else:
                correlation, pvalue = spearmanr(
                    artist_df[feature],
                    artist_df[
                        TARGET_CONTINUOUS
                    ],
                )

            duration_rows.append(
                {
                    "group_type":
                        "artist",

                    "group_value":
                        artist,

                    "duration_feature":
                        feature,

                    "rows":
                        len(artist_df),

                    "spearman_correlation":
                        (
                            float(correlation)
                            if not np.isnan(
                                correlation
                            )
                            else np.nan
                        ),

                    "spearman_pvalue":
                        (
                            float(pvalue)
                            if not np.isnan(
                                pvalue
                            )
                            else np.nan
                        ),
                }
            )

    duration_results = pd.DataFrame(
        duration_rows
    )

    OUTPUT_BINARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    binary_results.to_csv(
        OUTPUT_BINARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    duration_results.to_csv(
        OUTPUT_DURATION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    artist_results.to_csv(
        OUTPUT_ARTIST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== STRATEGY SIGNAL DIRECTION AUDIT V1 ===\n"
    )

    print(
        "=== GLOBAL AND CATEGORY BINARY SIGNALS ===\n"
    )

    print(
        binary_results
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== ARTIST-LEVEL BINARY SIGNALS ===\n"
    )

    print(
        artist_results
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== DURATION DIRECTION AUDIT ===\n"
    )

    print(
        duration_results
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== GLOBAL SIGNAL DIRECTION SUMMARY ===\n"
    )

    global_binary = binary_results[
        binary_results[
            "group_type"
        ] == "global"
    ]

    print(
        global_binary[
            [
                "signal",
                "n_signal_0",
                "n_signal_1",
                "high_or_top_rate_difference",
                "median_performance_difference",
                "mann_whitney_pvalue",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nImportant interpretation:"
    )

    print(
        "Permutation importance identifies "
        "predictive dependence, not effect direction."
    )

    print(
        "Binary rate differences and median "
        "performance differences provide "
        "descriptive direction only."
    )

    print(
        "Within-category and within-artist "
        "patterns must be checked for confounding."
    )

    print(
        "These analyses are observational and "
        "must not be interpreted causally."
    )

    print("\nSaved binary direction audit:")
    print(OUTPUT_BINARY_FILE)

    print("\nSaved duration direction audit:")
    print(OUTPUT_DURATION_FILE)

    print("\nSaved artist direction audit:")
    print(OUTPUT_ARTIST_FILE)


if __name__ == "__main__":
    main()
