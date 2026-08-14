from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trusted_needs_content_review_audit.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trusted_content_review_v1.csv"
)


PERFORMANCE_PATTERNS = [
    r"\blive\b",
    r"live performance",
    r"unplugged",
    r"اجرای زنده",
    r"زنده",
    r"کنسرت",
]

SUSPICIOUS_PATTERNS = [
    r"\bmashup\b",
    r"\bremix\b",
    r"همخوانی",
    r"\bclub version\b",
]


def contains_pattern(text, patterns):
    text = str(text).lower()

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def iso_duration_to_seconds(value):
    value = str(value)

    pattern = re.compile(
        r"PT"
        r"(?:(\d+)H)?"
        r"(?:(\d+)M)?"
        r"(?:(\d+)S)?"
    )

    match = pattern.fullmatch(value)

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return (
        hours * 3600
        + minutes * 60
        + seconds
    )


def classify_content(row):
    title = str(row["api_video_title"])
    source_type = str(row["source_type"])

    category_id = pd.to_numeric(
        row["category_id"],
        errors="coerce",
    )

    duration_seconds = row["duration_seconds"]

    if contains_pattern(
        title,
        PERFORMANCE_PATTERNS,
    ):
        return (
            "non_release_performance",
            "explicit_live_or_unplugged_title",
        )

    if contains_pattern(
        title,
        SUSPICIOUS_PATTERNS,
    ):
        return (
            "manual_review",
            "suspicious_collaboration_or_modified_content",
        )

    music_duration_ok = (
        pd.notna(duration_seconds)
        and 90 <= duration_seconds <= 900
    )

    if (
        source_type in ["topic", "vevo"]
        and category_id == 10
        and music_duration_ok
    ):
        return (
            "likely_release",
            "topic_or_vevo_music_duration_signal",
        )

    if (
        source_type
        in [
            "official_artist",
            "trusted_music_platform",
            "label_or_distributor",
        ]
        and category_id == 10
        and music_duration_ok
    ):
        return (
            "likely_release",
            "trusted_music_source_and_music_duration_signal",
        )

    return (
        "manual_review",
        "insufficient_content_evidence_after_trusted_source",
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    df["duration_seconds"] = (
        df["duration"]
        .apply(iso_duration_to_seconds)
    )

    decisions = df.apply(
        classify_content,
        axis=1,
        result_type="expand",
    )

    decisions.columns = [
        "content_review_status_v1",
        "content_review_reason_v1",
    ]

    df[
        [
            "content_review_status_v1",
            "content_review_reason_v1",
        ]
    ] = decisions

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== TRUSTED CONTENT REVIEW V1 ===\n"
    )

    print(
        df["content_review_status_v1"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== REVIEW STATUS BY ARTIST ===\n"
    )

    print(
        pd.crosstab(
            df["artist_name_fa"],
            df["content_review_status_v1"],
        ).to_string()
    )

    print(
        "\n=== REVIEW STATUS BY SOURCE TYPE ===\n"
    )

    print(
        pd.crosstab(
            df["source_type"],
            df["content_review_status_v1"],
        ).to_string()
    )

    print(
        "\n=== NON-RELEASE PERFORMANCE ===\n"
    )

    performance = df[
        df["content_review_status_v1"]
        == "non_release_performance"
    ]

    print(
        performance[
            [
                "artist_name_fa",
                "video_id",
                "api_video_title",
                "api_channel_title",
                "content_review_reason_v1",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== MANUAL REVIEW ===\n"
    )

    manual = df[
        df["content_review_status_v1"]
        == "manual_review"
    ]

    print(
        manual[
            [
                "artist_name_fa",
                "video_id",
                "api_video_title",
                "api_channel_title",
                "source_type",
                "category_id",
                "duration",
                "content_review_reason_v1",
            ]
        ].to_string(index=False)
    )

    print("\nRows:", len(df))

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
