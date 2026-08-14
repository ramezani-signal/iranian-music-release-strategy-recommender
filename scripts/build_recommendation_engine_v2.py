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
    / "artist_strategy_recommendations_v2.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendation_summary_v2.csv"
)

JSON_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendations_v2.json"
)

AUDIT_READY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_engine_v2_support_audit_ready.csv"
)


TARGET = "is_high_or_top"

MIN_ARTIST_RELEASES = 5
MIN_ARTIST_POSITIVES_FOR_PERSONALIZATION = 2

ENGINE_VERSION = "RE_V2"


def confidence_from_support(
    support_count,
    evidence_grade,
    scope,
):
    if support_count <= 0:
        return "insufficient"

    if scope.startswith("category_"):
        if support_count >= 8:
            return "medium"
        if support_count >= 4:
            return "low_medium"
        return "low"

    if evidence_grade == "B_context_dependent":
        if support_count >= 8:
            return "medium_high"
        if support_count >= 4:
            return "medium"
        if support_count >= 3:
            return "low_medium"
        return "low"

    if evidence_grade == "C_exploratory":
        if support_count >= 6:
            return "medium"
        if support_count >= 3:
            return "low_medium"
        return "low"

    return "low"


def append_row(
    rows,
    *,
    artist,
    category,
    signal,
    rule_type,
    recommendation_scope,
    evidence_grade,
    recommendation_strength,
    confidence,
    recommendation_status,
    support_count,
    support_definition,
    signal_total_count=0,
    positive_signal_count=0,
    negative_signal_count=0,
    reference_positive_count=0,
    reference_negative_count=0,
    reference_total_count=0,
    recommendation_text="",
    evidence_summary="",
    uncertainty_note="",
):
    rows.append(
        {
            "artist_name_fa": artist,
            "category": category,
            "signal": signal,
            "rule_type": rule_type,
            "recommendation_scope": recommendation_scope,
            "evidence_grade": evidence_grade,
            "recommendation_strength": recommendation_strength,
            "confidence": confidence,
            "recommendation_status": recommendation_status,
            "support_count": int(support_count),
            "support_definition": support_definition,
            "signal_total_count": int(signal_total_count),
            "positive_signal_count": int(positive_signal_count),
            "negative_signal_count": int(negative_signal_count),
            "reference_positive_count": int(reference_positive_count),
            "reference_negative_count": int(reference_negative_count),
            "reference_total_count": int(reference_total_count),
            "recommendation_text": recommendation_text,
            "evidence_summary": evidence_summary,
            "uncertainty_note": uncertainty_note,
            "recommendation_engine_version": ENGINE_VERSION,
        }
    )


def add_artist_evidence_status(
    rows,
    artist_df,
    artist,
    category,
):
    positive_count = int(
        artist_df[TARGET].sum()
    )

    total_count = len(artist_df)
    negative_count = total_count - positive_count

    if positive_count < 2:
        append_row(
            rows,
            artist=artist,
            category=category,
            signal="insufficient_artist_evidence",
            rule_type="coverage_status",
            recommendation_scope="artist_status",
            evidence_grade="D_insufficient",
            recommendation_strength="insufficient",
            confidence="insufficient",
            recommendation_status=(
                "insufficient_artist_evidence"
            ),
            support_count=positive_count,
            support_definition=(
                "number_of_artist_high_or_top_releases"
            ),
            reference_positive_count=positive_count,
            reference_negative_count=negative_count,
            reference_total_count=total_count,
            recommendation_text=(
                "شواهد اختصاصی این هنرمند برای تولید "
                "توصیه شخصی‌سازی‌شده کافی نیست."
            ),
            evidence_summary=(
                f"artist high_or_top releases="
                f"{positive_count}; "
                f"total releases={total_count}"
            ),
            uncertainty_note=(
                "در صورت نمایش پیشنهاد دسته‌ای، آن پیشنهاد "
                "جایگزین شواهد مستقیم هنرمند نیست."
            ),
        )


def add_music_video_signal(
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

    positive_signal_count = int(
        positive_df[signal].sum()
    )

    negative_signal_count = int(
        negative_df[signal].sum()
    )

    signal_total_count = (
        positive_signal_count
        + negative_signal_count
    )

    positive_reference_count = len(
        positive_df
    )

    negative_reference_count = len(
        negative_df
    )

    if (
        positive_reference_count < 2
        or signal_total_count < 2
        or negative_reference_count == 0
    ):
        return

    positive_rate = (
        positive_signal_count
        / positive_reference_count
    )

    negative_rate = (
        negative_signal_count
        / negative_reference_count
    )

    rate_difference = (
        positive_rate - negative_rate
    )

    if rate_difference > -0.20:
        return

    confidence = confidence_from_support(
        signal_total_count,
        evidence["evidence_grade"],
        "artist_specific",
    )

    append_row(
        rows,
        artist=artist,
        category=category,
        signal=signal,
        rule_type=evidence["rule_type"],
        recommendation_scope="artist_specific",
        evidence_grade=evidence[
            "evidence_grade"
        ],
        recommendation_strength="cautious",
        confidence=confidence,
        recommendation_status="active_recommendation",
        support_count=signal_total_count,
        support_definition=(
            "total_artist_music_video_observations"
        ),
        signal_total_count=signal_total_count,
        positive_signal_count=(
            positive_signal_count
        ),
        negative_signal_count=(
            negative_signal_count
        ),
        reference_positive_count=(
            positive_reference_count
        ),
        reference_negative_count=(
            negative_reference_count
        ),
        reference_total_count=len(artist_df),
        recommendation_text=(
            "قالب یا برچسب «Music Video» برای این هنرمند "
            "با احتیاط ارزیابی شود؛ در سابقه موجود، "
            "نمونه‌های این قالب کمتر در گروه عملکرد بالا "
            "قرار گرفته‌اند."
        ),
        evidence_summary=(
            f"music-video observations="
            f"{signal_total_count}; "
            f"positive music videos="
            f"{positive_signal_count}; "
            f"negative music videos="
            f"{negative_signal_count}; "
            f"positive-group rate="
            f"{positive_rate * 100:.2f}%; "
            f"negative-group rate="
            f"{negative_rate * 100:.2f}%"
        ),
        uncertainty_note=(
            "این نتیجه بر تعداد واقعی نمونه‌های Music Video "
            "همین هنرمند متکی است و رابطه علّی را اثبات نمی‌کند."
        ),
    )


def add_lyric_video_signal(
    rows,
    artist_df,
    category_df,
    artist,
    category,
    evidence,
):
    signal = "title_has_lyric_video"

    artist_positive = artist_df[
        artist_df[TARGET] == 1
    ]

    artist_negative = artist_df[
        artist_df[TARGET] == 0
    ]

    artist_positive_signal = int(
        artist_positive[signal].sum()
    )

    artist_negative_signal = int(
        artist_negative[signal].sum()
    )

    artist_signal_total = (
        artist_positive_signal
        + artist_negative_signal
    )

    if (
        len(artist_positive) >= 2
        and artist_signal_total >= 2
    ):
        positive_rate = (
            artist_positive_signal
            / len(artist_positive)
        )

        overall_rate = (
            artist_signal_total
            / len(artist_df)
        )

        if positive_rate - overall_rate >= 0.15:
            confidence = confidence_from_support(
                artist_positive_signal,
                evidence["evidence_grade"],
                "artist_specific_exploratory",
            )

            append_row(
                rows,
                artist=artist,
                category=category,
                signal=signal,
                rule_type=evidence["rule_type"],
                recommendation_scope=(
                    "artist_specific_exploratory"
                ),
                evidence_grade=evidence[
                    "evidence_grade"
                ],
                recommendation_strength=(
                    "exploratory"
                ),
                confidence=confidence,
                recommendation_status=(
                    "active_recommendation"
                ),
                support_count=(
                    artist_positive_signal
                ),
                support_definition=(
                    "number_of_positive_artist_"
                    "lyric_video_observations"
                ),
                signal_total_count=(
                    artist_signal_total
                ),
                positive_signal_count=(
                    artist_positive_signal
                ),
                negative_signal_count=(
                    artist_negative_signal
                ),
                reference_positive_count=len(
                    artist_positive
                ),
                reference_negative_count=len(
                    artist_negative
                ),
                reference_total_count=len(
                    artist_df
                ),
                recommendation_text=(
                    "قالب «Lyric Video» برای این هنرمند "
                    "می‌تواند به‌صورت یک آزمایش محدود "
                    "و قابل اندازه‌گیری بررسی شود."
                ),
                evidence_summary=(
                    f"positive lyric videos="
                    f"{artist_positive_signal}; "
                    f"negative lyric videos="
                    f"{artist_negative_signal}; "
                    f"total lyric videos="
                    f"{artist_signal_total}"
                ),
                uncertainty_note=(
                    "این پیشنهاد اکتشافی است و باید با "
                    "آزمون محدود و مقایسه عملکرد اجرا شود."
                ),
            )
            return

    category_positive = category_df[
        category_df[TARGET] == 1
    ]

    category_negative = category_df[
        category_df[TARGET] == 0
    ]

    category_positive_signal = int(
        category_positive[signal].sum()
    )

    category_negative_signal = int(
        category_negative[signal].sum()
    )

    category_signal_total = (
        category_positive_signal
        + category_negative_signal
    )

    if (
        len(category_positive) < 3
        or category_signal_total < 2
        or category_positive_signal < 2
    ):
        return

    positive_rate = (
        category_positive_signal
        / len(category_positive)
    )

    overall_rate = (
        category_signal_total
        / len(category_df)
    )

    if positive_rate - overall_rate < 0.15:
        return

    confidence = confidence_from_support(
        category_positive_signal,
        evidence["evidence_grade"],
        "category_exploratory_fallback",
    )

    append_row(
        rows,
        artist=artist,
        category=category,
        signal=signal,
        rule_type=evidence["rule_type"],
        recommendation_scope=(
            "category_exploratory_fallback"
        ),
        evidence_grade=evidence[
            "evidence_grade"
        ],
        recommendation_strength="exploratory",
        confidence=confidence,
        recommendation_status=(
            "category_fallback_recommendation"
        ),
        support_count=(
            category_positive_signal
        ),
        support_definition=(
            "number_of_positive_category_"
            "lyric_video_observations"
        ),
        signal_total_count=(
            category_signal_total
        ),
        positive_signal_count=(
            category_positive_signal
        ),
        negative_signal_count=(
            category_negative_signal
        ),
        reference_positive_count=len(
            category_positive
        ),
        reference_negative_count=len(
            category_negative
        ),
        reference_total_count=len(
            category_df
        ),
        recommendation_text=(
            "در این دسته موسیقایی، قالب «Lyric Video» "
            "در بخشی از آثار پربازده دیده شده است؛ "
            "برای این هنرمند فقط به‌صورت آزمایشی بررسی شود."
        ),
        evidence_summary=(
            f"positive category lyric videos="
            f"{category_positive_signal}; "
            f"negative category lyric videos="
            f"{category_negative_signal}; "
            f"total category lyric videos="
            f"{category_signal_total}"
        ),
        uncertainty_note=(
            "این پیشنهاد fallback دسته‌ای است و "
            "شواهد مستقیم هنرمند را جایگزین نمی‌کند."
        ),
    )


def add_duration_signal(
    rows,
    artist_df,
    artist,
    category,
    evidence,
):
    positive_df = artist_df[
        artist_df[TARGET] == 1
    ]

    support_count = len(positive_df)

    if (
        len(artist_df) < MIN_ARTIST_RELEASES
        or support_count < 2
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

    if support_count == 2:
        status = "historical_observation"
        strength = "descriptive"
        confidence = "low"
        text = (
            "در دو اثر پربازده موجود، مدت قطعه‌ها "
            f"بین {positive_df['duration_minutes'].min():.2f} "
            f"تا {positive_df['duration_minutes'].max():.2f} "
            "دقیقه بوده است. این مقدار فقط یک مشاهده تاریخی است."
        )
    else:
        status = "active_recommendation"
        strength = "cautious"
        confidence = confidence_from_support(
            support_count,
            evidence["evidence_grade"],
            "artist_specific",
        )
        text = (
            "برای برنامه‌ریزی آثار آینده، بازه تاریخی "
            f"آثار پربازده این هنرمند حدود {q25:.2f} "
            f"تا {q75:.2f} دقیقه و میانه آن "
            f"{median:.2f} دقیقه بوده است."
        )

    append_row(
        rows,
        artist=artist,
        category=category,
        signal="duration_minutes",
        rule_type=evidence["rule_type"],
        recommendation_scope="artist_specific",
        evidence_grade=evidence[
            "evidence_grade"
        ],
        recommendation_strength=strength,
        confidence=confidence,
        recommendation_status=status,
        support_count=support_count,
        support_definition=(
            "number_of_positive_artist_releases_"
            "with_duration"
        ),
        reference_positive_count=support_count,
        reference_negative_count=(
            len(artist_df) - support_count
        ),
        reference_total_count=len(artist_df),
        recommendation_text=text,
        evidence_summary=(
            f"positive duration observations="
            f"{support_count}; "
            f"IQR={q25:.2f}-{q75:.2f}; "
            f"median={median:.2f}"
        ),
        uncertainty_note=(
            "این بازه باید همراه با ساختار قطعه، سبک، "
            "هدف انتشار و اندازه نمونه تفسیر شود."
        ),
    )


def add_cadence_signal(
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

    support_count = len(positive_df)

    if support_count < 2:
        return

    values = positive_df[
        "days_since_previous_release"
    ]

    q25 = values.quantile(0.25)
    median = values.median()
    q75 = values.quantile(0.75)

    if support_count == 2:
        status = "historical_observation"
        strength = "descriptive"
        confidence = "low"
        text = (
            "در دو انتشار پربازده قابل مقایسه، "
            f"فاصله انتشار بین {values.min():.1f} "
            f"تا {values.max():.1f} روز بوده است. "
            "این مقدار فقط یک مشاهده تاریخی است."
        )
    else:
        status = "active_recommendation"
        strength = "exploratory"
        confidence = confidence_from_support(
            support_count,
            evidence["evidence_grade"],
            "artist_specific_exploratory",
        )
        text = (
            "فاصله انتشار آثار پربازده این هنرمند "
            f"عمدتاً بین {q25:.1f} تا {q75:.1f} روز "
            f"و میانه آن {median:.1f} روز بوده است."
        )

    append_row(
        rows,
        artist=artist,
        category=category,
        signal="days_since_previous_release",
        rule_type=evidence["rule_type"],
        recommendation_scope=(
            "artist_specific_exploratory"
        ),
        evidence_grade=evidence[
            "evidence_grade"
        ],
        recommendation_strength=strength,
        confidence=confidence,
        recommendation_status=status,
        support_count=support_count,
        support_definition=(
            "number_of_positive_artist_releases_"
            "with_valid_previous_release_gap"
        ),
        reference_positive_count=support_count,
        reference_negative_count=(
            len(artist_df) - support_count
        ),
        reference_total_count=len(artist_df),
        recommendation_text=text,
        evidence_summary=(
            f"positive cadence observations="
            f"{support_count}; "
            f"IQR={q25:.1f}-{q75:.1f}; "
            f"median={median:.1f}"
        ),
        uncertainty_note=(
            "فاصله انتشار ممکن است تحت تأثیر کمپین، "
            "فصل، بودجه، آماده‌بودن اثر و سیاست پلتفرم باشد."
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

    expected_signals = {
        "title_has_music_video",
        "title_has_lyric_video",
        "duration_minutes",
        "days_since_previous_release",
    }

    if set(evidence_map) != expected_signals:
        raise ValueError(
            "Unexpected allowed evidence signals: "
            f"{sorted(evidence_map)}"
        )

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    rows = []

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

        add_artist_evidence_status(
            rows,
            artist_df,
            artist,
            category,
        )

        add_music_video_signal(
            rows,
            artist_df,
            artist,
            category,
            evidence_map[
                "title_has_music_video"
            ],
        )

        add_lyric_video_signal(
            rows,
            artist_df,
            category_df,
            artist,
            category,
            evidence_map[
                "title_has_lyric_video"
            ],
        )

        add_duration_signal(
            rows,
            artist_df,
            artist,
            category,
            evidence_map[
                "duration_minutes"
            ],
        )

        add_cadence_signal(
            rows,
            artist_df,
            artist,
            category,
            evidence_map[
                "days_since_previous_release"
            ],
        )

    recommendations = pd.DataFrame(rows)

    if recommendations.empty:
        raise ValueError(
            "No Recommendation Engine V2 rows generated."
        )

    input_artists = set(
        df["artist_name_fa"].unique()
    )

    output_artists = set(
        recommendations[
            "artist_name_fa"
        ].unique()
    )

    if input_artists != output_artists:
        missing = sorted(
            input_artists - output_artists
        )

        raise ValueError(
            "Artists missing from V2 output: "
            f"{missing}"
        )

    recommendations = recommendations.sort_values(
        [
            "artist_name_fa",
            "recommendation_status",
            "signal",
        ]
    )

    recommendations.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    recommendations.to_csv(
        AUDIT_READY_FILE,
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
            output_row_count=(
                "signal",
                "count",
            ),
            active_recommendation_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "active_recommendation"
                    ).sum()
                ),
            ),
            category_fallback_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "category_fallback_recommendation"
                    ).sum()
                ),
            ),
            historical_observation_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "historical_observation"
                    ).sum()
                ),
            ),
            insufficient_evidence_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "insufficient_artist_evidence"
                    ).sum()
                ),
            ),
            signals=(
                "signal",
                lambda values: " | ".join(
                    sorted(set(values))
                ),
            ),
            confidence_levels=(
                "confidence",
                lambda values: " | ".join(
                    sorted(set(values))
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
            ENGINE_VERSION,
        "methodological_note": (
            "All recommendations are observational, "
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
                "artist_name_fa": artist,
                "category": category,
                "outputs": group.to_dict(
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
        "\n=== RECOMMENDATION ENGINE V2 ===\n"
    )

    print(
        "Input artists:",
        len(input_artists),
    )

    print(
        "Output artists:",
        len(output_artists),
    )

    print(
        "Total output rows:",
        len(recommendations),
    )

    print(
        "\nRecommendation status counts:\n"
    )

    print(
        recommendations[
            "recommendation_status"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nRows by signal:\n"
    )

    print(
        recommendations["signal"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRows by confidence:\n"
    )

    print(
        recommendations["confidence"]
        .value_counts()
        .to_string()
    )

    print(
        "\nSupport definitions:\n"
    )

    print(
        recommendations[
            "support_definition"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== ARTIST COVERAGE SUMMARY ===\n"
    )

    print(
        summary
        .sort_values(
            "artist_name_fa"
        )
        .to_string(index=False)
    )

    print(
        "\n=== COMPLETE V2 OUTPUT ===\n"
    )

    display_columns = [
        "artist_name_fa",
        "category",
        "signal",
        "recommendation_status",
        "recommendation_scope",
        "evidence_grade",
        "confidence",
        "support_count",
        "support_definition",
        "signal_total_count",
        "positive_signal_count",
        "negative_signal_count",
        "reference_positive_count",
        "reference_total_count",
        "recommendation_text",
        "uncertainty_note",
    ]

    print(
        recommendations[
            display_columns
        ]
        .to_string(index=False)
    )

    print("\nSaved V2 recommendations:")
    print(OUTPUT_FILE)

    print("\nSaved V2 summary:")
    print(SUMMARY_FILE)

    print("\nSaved V2 JSON:")
    print(JSON_OUTPUT_FILE)

    print("\nSaved audit-ready output:")
    print(AUDIT_READY_FILE)


if __name__ == "__main__":
    main()
