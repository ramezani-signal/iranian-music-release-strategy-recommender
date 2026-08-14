import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "artist_video_candidates_enriched.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "artist_video_candidates_screened_v2.csv"


def iso8601_duration_to_seconds(value):
    if pd.isna(value):
        return None

    pattern = re.compile(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    )

    match = pattern.fullmatch(str(value))

    if not match:
        return None

    hours, minutes, seconds = match.groups()

    return (
        int(hours or 0) * 3600
        + int(minutes or 0) * 60
        + int(seconds or 0)
    )


NON_RELEASE_PATTERNS = [
    r"\breaction\b", r"ری.?اکشن", r"واکنش",
    r"مصاحبه", r"گفت.?وگو", r"خبر", r"فوری",
    r"دستگیری", r"بازداشت", r"صحبت های", r"صحبت‌های",
    r"\bteaser\b", r"coming soon", r"تریلر",
    r"پادکست", r"\bpodcast\b",
    r"\btop\s*\d+\b", r"best songs", r"top songs",
    r"بهترین آهنگ", r"گلچین", r"\bmix\b", r"میکس",
    r"\bmashup\b", r"ریمیکس", r"\bremix\b",
    r"\bmedley\b", r"مدلی",
    r"\bconcert\b", r"کنسرت", r"festival", r"فستیوال",
    r"نقد",
]

REVIEW_PATTERNS = [
    r"live performance",
    r"live in concert",
    r"\blive\b",
    r"اجرای زنده",
    r"لایو",
    r"\bunplugged\b",
    r"acoustic",
]

RELEASE_PATTERNS = [
    r"official video",
    r"official music video",
    r"music video",
    r"lyric video",
    r"official audio",
]


def matched_patterns(text, patterns):
    text = str(text).lower()
    return [
        pattern
        for pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def classify_row(row):
    title = str(row.get("api_video_title", ""))
    duration_seconds = row.get("duration_seconds")
    category_id = row.get("category_id")
    licensed = row.get("licensed_content")

    non_release = matched_patterns(title, NON_RELEASE_PATTERNS)
    review = matched_patterns(title, REVIEW_PATTERNS)
    release = matched_patterns(title, RELEASE_PATTERNS)

    reasons = []

    if duration_seconds is not None:
        if duration_seconds < 90:
            return "likely_non_release", "too_short_under_90_seconds"

        if duration_seconds > 900:
            return "likely_non_release", "too_long_over_15_minutes"

    if non_release:
        return "likely_non_release", "non_release_title_pattern"

    if review:
        return "needs_review", "review_title_pattern_live_or_unplugged"

    if release and category_id == 10:
        reasons.extend(["release_title_pattern", "music_category"])
        if licensed is True:
            reasons.append("licensed_content")
        return "likely_release", " | ".join(reasons)

    if category_id == 10 and licensed is True:
        return "needs_review", "music_category_and_licensed_but_no_release_pattern"

    return "needs_review", "insufficient_evidence"


def main():
    df = pd.read_csv(INPUT_FILE)

    df["duration_seconds"] = df["duration"].apply(
        iso8601_duration_to_seconds
    )

    classifications = df.apply(
        classify_row,
        axis=1,
        result_type="expand",
    )

    classifications.columns = [
        "screening_status_v2",
        "screening_reason_v2",
    ]

    df = pd.concat([df, classifications], axis=1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== V2 Screening status counts ===\n")
    print(df["screening_status_v2"].value_counts().to_string())

    print("\n=== V2 Screening by artist ===\n")
    print(
        pd.crosstab(
            df["artist_name_fa"],
            df["screening_status_v2"],
        ).to_string()
    )

    print("\nSaved V2 screened candidates to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
