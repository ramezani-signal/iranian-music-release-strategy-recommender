from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v1_timing.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v2_format.csv"
)


FORMAT_PATTERNS = {
    "title_has_official": [
        r"\bofficial\b",
        r"رسمی",
    ],
    "title_has_music_video": [
        r"\bmusic video\b",
        r"موزیک\s*وید",
    ],
    "title_has_lyric_video": [
        r"\blyric video\b",
        r"\blyrics video\b",
        r"متن آهنگ",
        r"با\s*متن",
    ],
    "title_has_audio": [
        r"\bofficial audio\b",
        r"\baudio\b",
    ],
    "title_has_visual": [
        r"\bvisualizer\b",
        r"\bvisual\b",
    ],
    "title_has_clip": [
        r"\bclip\b",
        r"کلیپ",
    ],
    "title_has_soundtrack": [
        r"تیتراژ",
        r"\bsoundtrack\b",
        r"\bost\b",
    ],
    "title_has_alternative_version": [
        r"\balternative version\b",
        r"\bversion\b",
        r"نسخه",
    ],
    "title_has_collaboration": [
        r"\bfeat\.?\b",
        r"\bft\.?\b",
        r"\bfeaturing\b",
        r"\s+x\s+",
        r"\s*&\s*",
        r"\s+and\s+",
        r"همراه",
        r"با صدای",
    ],
}


SOURCE_TYPES = [
    "official_artist",
    "trusted_music_platform",
    "vevo",
    "label_or_distributor",
    "topic",
]


def contains_any_pattern(text, patterns):
    value = str(text).lower()

    return int(
        any(
            re.search(
                pattern,
                value,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )
    )


def contains_persian(text):
    return int(
        bool(
            re.search(
                r"[\u0600-\u06FF]",
                str(text),
            )
        )
    )


def contains_latin(text):
    return int(
        bool(
            re.search(
                r"[A-Za-z]",
                str(text),
            )
        )
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

    title_col = "api_video_title"

    if title_col not in df.columns:
        raise ValueError(
            f"Missing title column: {title_col}"
        )

    title_text = (
        df[title_col]
        .fillna("")
        .astype(str)
    )

    for feature_name, patterns in (
        FORMAT_PATTERNS.items()
    ):
        df[feature_name] = title_text.apply(
            lambda value: contains_any_pattern(
                value,
                patterns,
            )
        )

    df["title_contains_persian"] = (
        title_text.apply(contains_persian)
    )

    df["title_contains_latin"] = (
        title_text.apply(contains_latin)
    )

    df["title_is_bilingual"] = (
        (
            df["title_contains_persian"] == 1
        )
        & (
            df["title_contains_latin"] == 1
        )
    ).astype(int)

    df["title_character_count"] = (
        title_text.str.len()
    )

    df["title_word_count"] = (
        title_text
        .str.split()
        .str.len()
    )

    df["title_has_separator"] = (
        title_text
        .str.contains(
            r"[-–—|()]",
            regex=True,
        )
        .astype(int)
    )

    # یک قالب اصلی و یکتا برای تحلیل
    conditions = [
        df["title_has_lyric_video"] == 1,
        df["title_has_music_video"] == 1,
        df["title_has_audio"] == 1,
        df["title_has_visual"] == 1,
        df["title_has_clip"] == 1,
    ]

    choices = [
        "lyric_video",
        "music_video",
        "audio",
        "visual",
        "clip",
    ]

    df["release_format_primary"] = np.select(
        conditions,
        choices,
        default="unspecified",
    )

    for source_type in SOURCE_TYPES:
        column_name = (
            "source_type_"
            + source_type
        )

        df[column_name] = (
            df["source_type"]
            == source_type
        ).astype(int)

    format_dummies = pd.get_dummies(
        df["release_format_primary"],
        prefix="format",
        dtype=int,
    )

    df = pd.concat(
        [
            df,
            format_dummies,
        ],
        axis=1,
    )

    binary_columns = (
        list(FORMAT_PATTERNS.keys())
        + [
            "title_contains_persian",
            "title_contains_latin",
            "title_is_bilingual",
            "title_has_separator",
        ]
        + [
            "source_type_" + value
            for value in SOURCE_TYPES
        ]
        + format_dummies.columns.tolist()
    )

    invalid_binary_columns = [
        column
        for column in binary_columns
        if not set(
            df[column]
            .dropna()
            .unique()
        ).issubset({0, 1})
    ]

    if invalid_binary_columns:
        raise ValueError(
            "Invalid binary values in: "
            f"{invalid_binary_columns}"
        )

    check_columns = (
        binary_columns
        + [
            "title_character_count",
            "title_word_count",
        ]
    )

    if df[check_columns].isna().sum().sum() != 0:
        raise ValueError(
            "Missing format feature values."
        )

    if not np.isfinite(
        df[check_columns]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite format feature values."
        )

    df["strategy_format_version"] = (
        "ST_FORMAT_V1"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RELEASE FORMAT FEATURES V1 ===\n"
    )

    print("Rows:", len(df))
    print(
        "New binary feature count:",
        len(binary_columns),
    )

    print(
        "\nPrimary format counts:\n"
    )

    print(
        df["release_format_primary"]
        .value_counts()
        .to_string()
    )

    print(
        "\nTitle metadata counts:\n"
    )

    title_feature_columns = (
        list(FORMAT_PATTERNS.keys())
        + [
            "title_contains_persian",
            "title_contains_latin",
            "title_is_bilingual",
            "title_has_separator",
        ]
    )

    print(
        df[title_feature_columns]
        .sum()
        .sort_values(
            ascending=False
        )
        .to_string()
    )

    print(
        "\nSource type counts:\n"
    )

    print(
        df["source_type"]
        .value_counts()
        .to_string()
    )

    print(
        "\nTitle length summary:\n"
    )

    print(
        df[
            [
                "title_character_count",
                "title_word_count",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\nPerformance by primary format:\n"
    )

    print(
        df.groupby(
            "release_format_primary"
        )[
            "performance_percentile_final"
        ]
        .agg(
            [
                "count",
                "median",
                "mean",
            ]
        )
        .round(2)
        .sort_values(
            "median",
            ascending=False,
        )
        .to_string()
    )

    print(
        "\nPerformance by source type:\n"
    )

    print(
        df.groupby("source_type")[
            "performance_percentile_final"
        ]
        .agg(
            [
                "count",
                "median",
                "mean",
            ]
        )
        .round(2)
        .sort_values(
            "median",
            ascending=False,
        )
        .to_string()
    )

    print(
        "\nHigh-or-top rate by primary format:\n"
    )

    format_target = (
        df.groupby(
            "release_format_primary"
        )
        .agg(
            count=(
                "video_id",
                "count",
            ),
            high_or_top_count=(
                "is_high_or_top",
                "sum",
            ),
            high_or_top_rate=(
                "is_high_or_top",
                "mean",
            ),
        )
    )

    format_target[
        "high_or_top_rate"
    ] = (
        format_target[
            "high_or_top_rate"
        ]
        * 100
    ).round(2)

    print(
        format_target
        .sort_values(
            "high_or_top_rate",
            ascending=False,
        )
        .to_string()
    )

    print(
        "\nPrimary format by category:\n"
    )

    print(
        pd.crosstab(
            df["category"],
            df["release_format_primary"],
        ).to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
