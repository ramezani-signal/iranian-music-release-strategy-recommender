import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "artist_video_candidates_enriched.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened.csv"
)


NON_RELEASE_PATTERNS = [
    r"\breaction\b",
    r"ری.?اکشن",
    r"واکنش",
    r"مصاحبه",
    r"گفت.?وگو",
    r"خبر",
    r"فوری",
    r"دستگیری",
    r"بازداشت",
    r"صحبت های",
    r"صحبت‌های",
    r"\bteaser\b",
    r"coming soon",
    r"تریلر",
    r"پادکست",
    r"\bpodcast\b",
    r"\btop\s*\d+\b",
    r"best songs",
    r"top songs",
    r"بهترین آهنگ",
    r"گلچین",
    r"\bmix\b",
    r"میکس",
    r"\bmashup\b",
    r"ریمیکس",
    r"\bremix\b",
]


RELEASE_PATTERNS = [
    r"official video",
    r"official music video",
    r"music video",
    r"lyric video",
    r"official audio",
]


def contains_pattern(text, patterns):
    text = str(text).lower()

    matches = []

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)

    return matches


def classify_row(row):
    title = str(row.get("api_video_title", ""))
    duration = str(row.get("duration", ""))
    category_id = row.get("category_id")
    licensed = row.get("licensed_content")

    non_release_matches = contains_pattern(
        title,
        NON_RELEASE_PATTERNS,
    )

    release_matches = contains_pattern(
        title,
        RELEASE_PATTERNS,
    )

    reasons = []

    if non_release_matches:
        reasons.append(
            "non_release_title_pattern"
        )

        return (
            "likely_non_release",
            " | ".join(reasons),
        )

    if release_matches:
        reasons.append(
            "release_title_pattern"
        )

    if category_id == 10:
        reasons.append(
            "music_category"
        )

    if licensed is True:
        reasons.append(
            "licensed_content"
        )

    if (
        release_matches
        and category_id == 10
    ):
        return (
            "likely_release",
            " | ".join(reasons),
        )

    if (
        category_id == 10
        and licensed is True
    ):
        return (
            "likely_release",
            " | ".join(reasons),
        )

    return (
        "needs_review",
        " | ".join(reasons)
        if reasons
        else "insufficient_evidence",
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    classifications = df.apply(
        classify_row,
        axis=1,
        result_type="expand",
    )

    classifications.columns = [
        "screening_status",
        "screening_reason",
    ]

    df = pd.concat(
        [
            df,
            classifications,
        ],
        axis=1,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Screening status counts ===\n")

    print(
        df["screening_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== Screening by artist ===\n"
    )

    summary = pd.crosstab(
        df["artist_name_fa"],
        df["screening_status"],
    )

    print(summary.to_string())

    print(
        "\nSaved screened candidates to:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
