from pathlib import Path
import json

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
    / "release_strategy_feature_matrix_v1.csv"
)

CONTRACT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_feature_contract_v1.json"
)


CORE_STRATEGY_FEATURES = [
    "duration_minutes",
    "release_hour_tehran",
    "is_iran_weekend_release",
    "days_since_previous_release",
    "title_character_count",
    "title_word_count",
    "title_has_official",
    "title_has_music_video",
    "title_has_lyric_video",
    "title_has_collaboration",
    "title_contains_persian",
    "title_is_bilingual",
]


CONTEXT_FEATURES = [
    "category_classic_fusion",
    "category_pop",
    "category_rap_hiphop",
]


EXPERIMENTAL_FEATURES = [
    "title_has_visual",
    "title_has_clip",
    "title_has_soundtrack",
    "title_has_alternative_version",
    "title_has_separator",
]


TARGET_COLUMNS = [
    "performance_index_raw_final",
    "performance_percentile_final",
    "performance_tier_final",
    "is_top_performer",
    "is_high_or_top",
]


IDENTIFIER_COLUMNS = [
    "record_id",
    "artist_id",
    "artist_name_fa",
    "category",
    "video_id",
    "api_video_title",
    "api_channel_title",
    "published_at",
]


def robust_zscore(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = values.median()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return (
        values - median
    ) / iqr


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
        IDENTIFIER_COLUMNS
        + CORE_STRATEGY_FEATURES
        + CONTEXT_FEATURES
        + EXPERIMENTAL_FEATURES
        + TARGET_COLUMNS
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

    # Missing previous-release gap is structurally expected
    # for the first observed release of each artist.
    df[
        "days_since_previous_release_missing"
    ] = (
        df["days_since_previous_release"]
        .isna()
        .astype(int)
    )

    category_gap_median = (
        df.groupby("category")[
            "days_since_previous_release"
        ]
        .transform("median")
    )

    global_gap_median = (
        df["days_since_previous_release"]
        .median()
    )

    df[
        "days_since_previous_release_imputed"
    ] = (
        df["days_since_previous_release"]
        .fillna(category_gap_median)
        .fillna(global_gap_median)
    )

    CONTINUOUS_MODEL_FEATURES = [
        "duration_minutes",
        "release_hour_tehran",
        "days_since_previous_release_imputed",
        "title_character_count",
        "title_word_count",
    ]

    ROBUST_FEATURES = []

    for feature in CONTINUOUS_MODEL_FEATURES:
        robust_column = (
            "strategy_rz_" + feature
        )

        df[robust_column] = robust_zscore(
            df[feature]
        )

        ROBUST_FEATURES.append(
            robust_column
        )

    BINARY_MODEL_FEATURES = [
        "is_iran_weekend_release",
        "days_since_previous_release_missing",
        "title_has_official",
        "title_has_music_video",
        "title_has_lyric_video",
        "title_has_collaboration",
        "title_contains_persian",
        "title_is_bilingual",
    ]

    MODEL_FEATURES = (
        ROBUST_FEATURES
        + BINARY_MODEL_FEATURES
        + CONTEXT_FEATURES
    )

    overlap = (
        set(MODEL_FEATURES)
        & set(TARGET_COLUMNS)
    )

    if overlap:
        raise ValueError(
            "Target leakage detected: "
            f"{sorted(overlap)}"
        )

    if df[MODEL_FEATURES].isna().sum().sum() != 0:
        raise ValueError(
            "Missing values in model features."
        )

    numeric_matrix = (
        df[MODEL_FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        numeric_matrix
    ).all():
        raise ValueError(
            "Infinite values in model features."
        )

    binary_features = (
        BINARY_MODEL_FEATURES
        + CONTEXT_FEATURES
    )

    invalid_binary = [
        column
        for column in binary_features
        if not set(
            df[column]
            .dropna()
            .unique()
        ).issubset({0, 1})
    ]

    if invalid_binary:
        raise ValueError(
            "Invalid binary features: "
            f"{invalid_binary}"
        )

    category_row_sum = (
        df[CONTEXT_FEATURES]
        .sum(axis=1)
    )

    if not (
        category_row_sum == 1
    ).all():
        raise ValueError(
            "Category dummy contract violated."
        )

    df[
        "strategy_feature_matrix_version"
    ] = "SFM_V1"

    output_columns = (
        IDENTIFIER_COLUMNS
        + CORE_STRATEGY_FEATURES
        + [
            "days_since_previous_release_missing",
            "days_since_previous_release_imputed",
        ]
        + ROBUST_FEATURES
        + BINARY_MODEL_FEATURES
        + CONTEXT_FEATURES
        + EXPERIMENTAL_FEATURES
        + TARGET_COLUMNS
        + [
            "strategy_feature_matrix_version",
        ]
    )

    output_columns = list(
        dict.fromkeys(output_columns)
    )

    output_df = df[
        output_columns
    ].copy()

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    contract = {
        "contract_version":
            "STRATEGY_FEATURE_CONTRACT_V1",
        "feature_matrix_version":
            "SFM_V1",
        "row_count":
            int(len(output_df)),
        "core_strategy_features":
            CORE_STRATEGY_FEATURES,
        "continuous_model_features":
            CONTINUOUS_MODEL_FEATURES,
        "robust_model_features":
            ROBUST_FEATURES,
        "binary_model_features":
            BINARY_MODEL_FEATURES,
        "context_features":
            CONTEXT_FEATURES,
        "experimental_features":
            EXPERIMENTAL_FEATURES,
        "model_features":
            MODEL_FEATURES,
        "target_columns":
            TARGET_COLUMNS,
        "notes": {
            "days_since_previous_release":
                "First observed release per artist is structurally missing.",
            "gap_imputation":
                "Category median, then global median fallback.",
            "experimental_features":
                "Retained for analysis but excluded from baseline model due to sparsity or unstable evidence.",
            "causal_interpretation":
                "Feature associations are observational and must not be interpreted as causal effects.",
        },
    }

    CONTRACT_FILE.write_text(
        json.dumps(
            contract,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== STRATEGY FEATURE CONTRACT V1 ===\n"
    )

    print("Rows:", len(output_df))
    print(
        "Output columns:",
        len(output_df.columns),
    )

    print(
        "\nCore strategy features:"
    )

    for column in CORE_STRATEGY_FEATURES:
        print(column)

    print(
        "\nRobust model features:"
    )

    for column in ROBUST_FEATURES:
        print(column)

    print(
        "\nBinary model features:"
    )

    for column in BINARY_MODEL_FEATURES:
        print(column)

    print(
        "\nContext features:"
    )

    for column in CONTEXT_FEATURES:
        print(column)

    print(
        "\nExperimental features excluded from baseline model:"
    )

    for column in EXPERIMENTAL_FEATURES:
        print(column)

    print(
        "\nFinal baseline model feature count:",
        len(MODEL_FEATURES),
    )

    print(
        "\nModel features:"
    )

    for column in MODEL_FEATURES:
        print(column)

    print(
        "\nMissing values in model features:"
    )

    print(
        output_df[MODEL_FEATURES]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nDays-since-previous-release missing indicator counts:"
    )

    print(
        output_df[
            "days_since_previous_release_missing"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nImputed release gap summary:"
    )

    print(
        output_df[
            "days_since_previous_release_imputed"
        ]
        .describe()
        .round(4)
        .to_string()
    )

    print(
        "\nCategory dummy row-sum counts:"
    )

    print(
        output_df[
            CONTEXT_FEATURES
        ]
        .sum(axis=1)
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTarget counts:"
    )

    print(
        "\nis_high_or_top:"
    )

    print(
        output_df[
            "is_high_or_top"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nis_top_performer:"
    )

    print(
        output_df[
            "is_top_performer"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nSaved feature matrix:"
    )

    print(OUTPUT_FILE)

    print(
        "\nSaved feature contract:"
    )

    print(CONTRACT_FILE)


if __name__ == "__main__":
    main()
