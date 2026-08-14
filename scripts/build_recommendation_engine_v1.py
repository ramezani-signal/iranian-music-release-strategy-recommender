from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v2_format.csv"
)

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_evidence_registry_v1.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendations_v1.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendation_summary_v1.csv"
)

JSON_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendations_v1.json"
)


TARGET = "is_high_or_top"

MIN_ARTIST_RELEASES = 5
MIN_ARTIST_POSITIVE_RELEASES = 2
MIN_BINARY_SIGNAL_ON_COUNT = 2
MIN_CATEGORY_POSITIVE_RELEASES = 3


def confidence_from_support(
    positive_count,
    total_count,
    evidence_grade,
):
    if total_count <= 0:
        return "insufficient"

    if (
        evidence_grade
        == "B_context_dependent"
        and positive_count >= 4
        and total_count >= 8
    ):
        return "medium_high"

    if (
        positive_count >= 3
        and total_count >= 6
    ):
        return "medium"

    if positive_count >= 2:
        return "low_medium"

    return "low"


def safe_percent(value):
    if pd.isna(value):
        return np.nan

    return round(
        float(value) * 100,
        2,
    )


def append_recommendation(
    rows,
    artist,
    category,
    signal,
    rule_type,
    scope,
    evidence_grade,
    strength,
    confidence,
    support_count,
    comparison_count,
    recommendation_text,
    evidence_summary,
    uncertainty_note,
):
    rows.append(
        {
            "artist_name_fa":
                artist,

            "category":
                category,

            "signal":
                signal,

            "rule_type":
                rule_type,

            "recommendation_scope":
                scope,

            "evidence_grade":
                evidence_grade,

            "recommendation_strength":
                strength,

            "confidence":
                confidence,

            "support_count":
                int(support_count),

            "comparison_count":
                int(comparison_count),

            "recommendation_text":
                recommendation_text,

            "evidence_summary":
                evidence_summary,

            "uncertainty_note":
                uncertainty_note,

            "recommendation_engine_version":
                "RE_V1",
        }
    )


def build_music_video_recommendation(
    rows,
    artist_df,
    artist,
    category,
    evidence,
):
    signal = "title_has_music_video"

    positive_df = artist_df[
        artist_df[TARGET] == 1
    ]

    negative_df = artist_df[
        artist_df[TARGET] == 0
    ]

    positive_on = int(
        positive_df[signal].sum()
    )

    positive_total = len(positive_df)

    negative_on = int(
        negative_df[signal].sum()
    )

    negative_total = len(negative_df)

    if (
        positive_total < MIN_ARTIST_POSITIVE_RELEASES
        or artist_df[signal].sum()
        < MIN_BINARY_SIGNAL_ON_COUNT
    ):
        return

    positive_rate = (
        positive_on / positive_total
        if positive_total > 0
        else np.nan
    )

    negative_rate = (
        negative_on / negative_total
        if negative_total > 0
        else np.nan
    )

    if pd.isna(negative_rate):
        return

    rate_difference = (
        positive_rate - negative_rate
    )

    if rate_difference <= -0.20:
        confidence = confidence_from_support(
            positive_total,
            len(artist_df),
            evidence["evidence_grade"],
        )

        append_recommendation(
            rows=rows,
            artist=artist,
            category=category,
            signal=signal,
            rule_type=evidence["rule_type"],
            scope="artist_specific",
            evidence_grade=evidence["evidence_grade"],
            strength="cautious",
            confidence=confidence,
            support_count=positive_total,
            comparison_count=len(artist_df),
            recommendation_text=(
                "برچسب یا قالب «Music Video» برای این هنرمند "
                "با احتیاط استفاده شود؛ در داده تاریخی همین هنرمند، "
                "این قالب در میان آثار پربازده سهم کمتری داشته است."
            ),
            evidence_summary=(
                f"high_or_top music-video rate="
                f"{safe_percent(positive_rate)}%; "
                f"other-release music-video rate="
                f"{safe_percent(negative_rate)}%"
            ),
            uncertainty_note=(
                "این رابطه مشاهده‌ای است و ممکن است ناشی از نوع اثر، "
                "بودجه تولید، زمان انتشار یا سیاست کانال باشد."
            ),
        )


def build_lyric_video_recommendation(
    rows,
    artist_df,
    category_df,
    artist,
    category,
    evidence,
):
    signal = "title_has_lyric_video"

    positive_df = artist_df[
        artist_df[TARGET] == 1
    ]

    positive_total = len(positive_df)
    artist_on_count = int(
        artist_df[signal].sum()
    )

    if (
        positive_total
        >= MIN_ARTIST_POSITIVE_RELEASES
        and artist_on_count
        >= MIN_BINARY_SIGNAL_ON_COUNT
    ):
        positive_rate = (
            positive_df[signal].mean()
        )

        overall_rate = (
            artist_df[signal].mean()
        )

        if (
            positive_rate
            - overall_rate
            >= 0.15
        ):
            confidence = confidence_from_support(
                positive_total,
                len(artist_df),
                evidence["evidence_grade"],
            )

            append_recommendation(
                rows=rows,
                artist=artist,
                category=category,
                signal=signal,
                rule_type=evidence["rule_type"],
                scope="artist_specific_exploratory",
                evidence_grade=evidence["evidence_grade"],
                strength="exploratory",
                confidence=confidence,
                support_count=positive_total,
                comparison_count=len(artist_df),
                recommendation_text=(
                    "آزمایش قالب «Lyric Video» برای این هنرمند "
                    "می‌تواند به‌عنوان یک گزینه اکتشافی در نظر گرفته شود."
                ),
                evidence_summary=(
                    f"high_or_top lyric-video rate="
                    f"{safe_percent(positive_rate)}%; "
                    f"overall artist lyric-video rate="
                    f"{safe_percent(overall_rate)}%"
                ),
                uncertainty_note=(
                    "تعداد نمونه‌ها محدود است و این پیشنهاد باید "
                    "به‌صورت آزمایش کنترل‌شده اجرا شود."
                ),
            )

            return

    category_positive = category_df[
        category_df[TARGET] == 1
    ]

    if (
        len(category_positive)
        >= MIN_CATEGORY_POSITIVE_RELEASES
        and category_df[signal].sum()
        >= MIN_BINARY_SIGNAL_ON_COUNT
    ):
        positive_rate = (
            category_positive[signal].mean()
        )

        overall_rate = (
            category_df[signal].mean()
        )

        if (
            positive_rate
            - overall_rate
            >= 0.15
        ):
            append_recommendation(
                rows=rows,
                artist=artist,
                category=category,
                signal=signal,
                rule_type=evidence["rule_type"],
                scope="category_exploratory_fallback",
                evidence_grade=evidence["evidence_grade"],
                strength="exploratory",
                confidence="low",
                support_count=len(category_positive),
                comparison_count=len(category_df),
                recommendation_text=(
                    "در این دسته موسیقایی، قالب «Lyric Video» "
                    "در میان برخی آثار پربازده بیشتر دیده شده است؛ "
                    "برای این هنرمند فقط به‌صورت آزمایشی بررسی شود."
                ),
                evidence_summary=(
                    f"category high_or_top lyric-video rate="
                    f"{safe_percent(positive_rate)}%; "
                    f"category overall rate="
                    f"{safe_percent(overall_rate)}%"
                ),
                uncertainty_note=(
                    "این پیشنهاد از الگوی دسته موسیقایی استخراج شده "
                    "و شواهد مستقیم هنرمند کافی نیست."
                ),
            )


def build_duration_recommendation(
    rows,
    artist_df,
    artist,
    category,
    evidence,
):
    positive_df = artist_df[
        artist_df[TARGET] == 1
    ]

    if (
        len(artist_df)
        < MIN_ARTIST_RELEASES
        or len(positive_df)
        < MIN_ARTIST_POSITIVE_RELEASES
    ):
        return

    q25 = positive_df[
        "duration_minutes"
    ].quantile(0.25)

    median = positive_df[
        "duration_minutes"
    ].median()

    q75 = positive_df[
        "duration_minutes"
    ].quantile(0.75)

    if any(
        pd.isna(value)
        for value in [
            q25,
            median,
            q75,
        ]
    ):
        return

    confidence = confidence_from_support(
        len(positive_df),
        len(artist_df),
        evidence["evidence_grade"],
    )

    append_recommendation(
        rows=rows,
        artist=artist,
        category=category,
        signal="duration_minutes",
        rule_type=evidence["rule_type"],
        scope="artist_specific",
        evidence_grade=evidence["evidence_grade"],
        strength="cautious",
        confidence=confidence,
        support_count=len(positive_df),
        comparison_count=len(artist_df),
        recommendation_text=(
            "برای برنامه‌ریزی آثار آینده، بازه مدت آثار پربازده "
            f"این هنرمند حدود {q25:.2f} تا {q75:.2f} دقیقه "
            f"و میانه آن {median:.2f} دقیقه بوده است."
        ),
        evidence_summary=(
            f"positive-release duration IQR="
            f"{q25:.2f}-{q75:.2f} minutes; "
            f"median={median:.2f}"
        ),
        uncertainty_note=(
            "این بازه نسخه‌برداری از گذشته نیست و باید همراه با "
            "ساختار قطعه، سبک و هدف انتشار تفسیر شود."
        ),
    )


def build_cadence_recommendation(
    rows,
    artist_df,
    artist,
    category,
    evidence,
):
    positive_df = artist_df[
        (
            artist_df[TARGET] == 1
        )
        & (
            artist_df[
                "days_since_previous_release"
            ].notna()
        )
    ]

    if len(positive_df) < 2:
        return

    q25 = positive_df[
        "days_since_previous_release"
    ].quantile(0.25)

    median = positive_df[
        "days_since_previous_release"
    ].median()

    q75 = positive_df[
        "days_since_previous_release"
    ].quantile(0.75)

    confidence = confidence_from_support(
        len(positive_df),
        len(artist_df),
        evidence["evidence_grade"],
    )

    append_recommendation(
        rows=rows,
        artist=artist,
        category=category,
        signal="days_since_previous_release",
        rule_type=evidence["rule_type"],
        scope="artist_specific_exploratory",
        evidence_grade=evidence["evidence_grade"],
        strength="exploratory",
        confidence=confidence,
        support_count=len(positive_df),
        comparison_count=len(artist_df),
        recommendation_text=(
            "فاصله انتشار آثار پربازده این هنرمند در داده موجود "
            f"عمدتاً بین {q25:.1f} تا {q75:.1f} روز بوده "
            f"و میانه آن {median:.1f} روز است."
        ),
        evidence_summary=(
            f"positive-release cadence IQR="
            f"{q25:.1f}-{q75:.1f} days; "
            f"median={median:.1f}"
        ),
        uncertainty_note=(
            "فاصله انتشار تحت تأثیر کمپین تبلیغاتی، فصل، "
            "آمادگی اثر و برنامه پلتفرم قرار دارد."
        ),
    )


def main():
    df = pd.read_csv(DATA_FILE)

    evidence_payload = json.loads(
        EVIDENCE_FILE.read_text(
            encoding="utf-8"
        )
    )

    evidence_map = {
        row["signal"]: row
        for row in evidence_payload["signals"]
        if row["allowed_in_engine"]
    }

    required_signals = {
        "title_has_music_video",
        "title_has_lyric_video",
        "duration_minutes",
        "days_since_previous_release",
    }

    if set(evidence_map) != required_signals:
        raise ValueError(
            "Unexpected allowed evidence signals: "
            f"{sorted(evidence_map)}"
        )

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    recommendation_rows = []

    for (
        artist,
        category,
    ), artist_df in df.groupby(
        [
            "artist_name_fa",
            "category",
        ]
    ):
        category_df = df[
            df["category"] == category
        ]

        build_music_video_recommendation(
            rows=recommendation_rows,
            artist_df=artist_df,
            artist=artist,
            category=category,
            evidence=evidence_map[
                "title_has_music_video"
            ],
        )

        build_lyric_video_recommendation(
            rows=recommendation_rows,
            artist_df=artist_df,
            category_df=category_df,
            artist=artist,
            category=category,
            evidence=evidence_map[
                "title_has_lyric_video"
            ],
        )

        build_duration_recommendation(
            rows=recommendation_rows,
            artist_df=artist_df,
            artist=artist,
            category=category,
            evidence=evidence_map[
                "duration_minutes"
            ],
        )

        build_cadence_recommendation(
            rows=recommendation_rows,
            artist_df=artist_df,
            artist=artist,
            category=category,
            evidence=evidence_map[
                "days_since_previous_release"
            ],
        )

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    if recommendations.empty:
        raise ValueError(
            "No recommendations were generated."
        )

    recommendations = recommendations.sort_values(
        [
            "artist_name_fa",
            "evidence_grade",
            "signal",
        ]
    )

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        recommendations.groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )
        .agg(
            recommendation_count=(
                "signal",
                "count",
            ),
            signals=(
                "signal",
                lambda x: " | ".join(
                    sorted(set(x))
                ),
            ),
            evidence_grades=(
                "evidence_grade",
                lambda x: " | ".join(
                    sorted(set(x))
                ),
            ),
            confidence_levels=(
                "confidence",
                lambda x: " | ".join(
                    sorted(set(x))
                ),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "recommendation_engine_version":
            "RE_V1",

        "methodological_note":
            (
                "Recommendations are observational, "
                "context-dependent and non-causal."
            ),

        "artists": [],
    }

    for (
        artist,
        category,
    ), group in recommendations.groupby(
        [
            "artist_name_fa",
            "category",
        ]
    ):
        payload["artists"].append(
            {
                "artist_name_fa":
                    artist,

                "category":
                    category,

                "recommendations":
                    group.to_dict(
                        orient="records"
                    ),
            }
        )

    JSON_OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== RECOMMENDATION ENGINE V1 ===\n"
    )

    print(
        "Artists in input:",
        df["artist_name_fa"].nunique(),
    )

    print(
        "Artists with recommendations:",
        recommendations[
            "artist_name_fa"
        ].nunique(),
    )

    print(
        "Total recommendations:",
        len(recommendations),
    )

    print(
        "\nRecommendations by signal:\n"
    )

    print(
        recommendations["signal"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommendations by evidence grade:\n"
    )

    print(
        recommendations["evidence_grade"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommendations by confidence:\n"
    )

    print(
        recommendations["confidence"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommendation count by artist:\n"
    )

    print(
        summary[
            [
                "artist_name_fa",
                "category",
                "recommendation_count",
                "signals",
                "evidence_grades",
                "confidence_levels",
            ]
        ]
        .sort_values(
            "recommendation_count",
            ascending=False,
        )
        .to_string(index=False)
    )

    print(
        "\n=== COMPLETE RECOMMENDATIONS ===\n"
    )

    print(
        recommendations[
            [
                "artist_name_fa",
                "category",
                "signal",
                "recommendation_scope",
                "evidence_grade",
                "confidence",
                "support_count",
                "comparison_count",
                "recommendation_text",
                "uncertainty_note",
            ]
        ]
        .to_string(index=False)
    )

    print("\nSaved recommendations CSV:")
    print(OUTPUT_FILE)

    print("\nSaved recommendation summary:")
    print(SUMMARY_FILE)

    print("\nSaved recommendations JSON:")
    print(JSON_OUTPUT_FILE)


if __name__ == "__main__":
    main()
