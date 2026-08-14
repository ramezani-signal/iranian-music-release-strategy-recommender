from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v2_format.csv"
)

RECOMMENDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_strategy_recommendations_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_engine_v1_quality_audit.csv"
)

ARTIST_COVERAGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_engine_v1_artist_coverage.csv"
)


TARGET = "is_high_or_top"


def main():
    data = pd.read_csv(DATA_FILE)
    recommendations = pd.read_csv(
        RECOMMENDATION_FILE
    )

    if len(data) != 115:
        raise ValueError(
            f"Expected 115 data rows, found {len(data)}."
        )

    if recommendations.empty:
        raise ValueError(
            "Recommendation file is empty."
        )

    audit_rows = []

    for _, recommendation in recommendations.iterrows():
        artist = recommendation["artist_name_fa"]
        category = recommendation["category"]
        signal = recommendation["signal"]
        scope = recommendation[
            "recommendation_scope"
        ]

        artist_df = data[
            data["artist_name_fa"] == artist
        ].copy()

        category_df = data[
            data["category"] == category
        ].copy()

        artist_positive = artist_df[
            artist_df[TARGET] == 1
        ]

        artist_negative = artist_df[
            artist_df[TARGET] == 0
        ]

        actual_support_count = 0
        actual_signal_count = 0
        actual_positive_signal_count = 0
        actual_negative_signal_count = 0

        issue_flags = []

        if signal in [
            "title_has_music_video",
            "title_has_lyric_video",
        ]:
            if scope.startswith("category_"):
                reference_df = category_df
            else:
                reference_df = artist_df

            actual_signal_count = int(
                reference_df[signal].sum()
            )

            actual_positive_signal_count = int(
                reference_df.loc[
                    reference_df[TARGET] == 1,
                    signal,
                ].sum()
            )

            actual_negative_signal_count = int(
                reference_df.loc[
                    reference_df[TARGET] == 0,
                    signal,
                ].sum()
            )

            actual_support_count = (
                actual_positive_signal_count
            )

            if actual_signal_count < 2:
                issue_flags.append(
                    "binary_signal_total_below_2"
                )

            if actual_positive_signal_count < 2:
                issue_flags.append(
                    "positive_signal_support_below_2"
                )

        elif signal == "duration_minutes":
            actual_support_count = len(
                artist_positive
            )

            if actual_support_count < 3:
                issue_flags.append(
                    "duration_range_support_below_3"
                )

        elif signal == "days_since_previous_release":
            actual_support_count = int(
                artist_positive[
                    "days_since_previous_release"
                ]
                .notna()
                .sum()
            )

            if actual_support_count < 3:
                issue_flags.append(
                    "cadence_range_support_below_3"
                )

        else:
            issue_flags.append(
                "unexpected_signal"
            )

        reported_support_count = int(
            recommendation["support_count"]
        )

        support_count_matches = int(
            reported_support_count
            == actual_support_count
        )

        if not support_count_matches:
            issue_flags.append(
                "reported_support_count_mismatch"
            )

        confidence = str(
            recommendation["confidence"]
        )

        if (
            actual_support_count < 3
            and confidence in [
                "medium",
                "medium_high",
            ]
        ):
            issue_flags.append(
                "confidence_too_high_for_support"
            )

        audit_rows.append(
            {
                "artist_name_fa":
                    artist,

                "category":
                    category,

                "signal":
                    signal,

                "recommendation_scope":
                    scope,

                "evidence_grade":
                    recommendation[
                        "evidence_grade"
                    ],

                "reported_confidence":
                    confidence,

                "reported_support_count":
                    reported_support_count,

                "actual_support_count":
                    actual_support_count,

                "actual_signal_count":
                    actual_signal_count,

                "actual_positive_signal_count":
                    actual_positive_signal_count,

                "actual_negative_signal_count":
                    actual_negative_signal_count,

                "support_count_matches":
                    support_count_matches,

                "issue_count":
                    len(issue_flags),

                "issue_flags":
                    " | ".join(issue_flags),

                "audit_status":
                    (
                        "pass"
                        if len(issue_flags) == 0
                        else "needs_revision"
                    ),
            }
        )

    audit = pd.DataFrame(audit_rows)

    input_artists = (
        data[
            [
                "artist_name_fa",
                "category",
            ]
        ]
        .drop_duplicates()
    )

    recommendation_counts = (
        recommendations
        .groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )
        .size()
        .reset_index(
            name="recommendation_count"
        )
    )

    coverage = input_artists.merge(
        recommendation_counts,
        on=[
            "artist_name_fa",
            "category",
        ],
        how="left",
    )

    coverage["recommendation_count"] = (
        coverage["recommendation_count"]
        .fillna(0)
        .astype(int)
    )

    positive_counts = (
        data.groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )[TARGET]
        .agg(
            release_count="count",
            positive_release_count="sum",
        )
        .reset_index()
    )

    coverage = coverage.merge(
        positive_counts,
        on=[
            "artist_name_fa",
            "category",
        ],
        how="left",
    )

    coverage["coverage_status"] = (
        coverage["recommendation_count"]
        .apply(
            lambda value:
                "recommendations_generated"
                if value > 0
                else "insufficient_artist_evidence"
        )
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        ARTIST_COVERAGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RECOMMENDATION ENGINE V1 "
        "QUALITY AUDIT ===\n"
    )

    print("Recommendations:", len(audit))

    print(
        "\nAudit status counts:\n"
    )

    print(
        audit["audit_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRecommendations with support mismatch:\n"
    )

    print(
        audit[
            "support_count_matches"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nIssue-flag counts:\n"
    )

    issue_counts = (
        audit["issue_flags"]
        .str.split(r" \| ")
        .explode()
    )

    issue_counts = issue_counts[
        issue_counts != ""
    ]

    if issue_counts.empty:
        print("None")
    else:
        print(
            issue_counts
            .value_counts()
            .to_string()
        )

    print(
        "\n=== COMPLETE QUALITY AUDIT ===\n"
    )

    print(
        audit[
            [
                "artist_name_fa",
                "signal",
                "recommendation_scope",
                "reported_confidence",
                "reported_support_count",
                "actual_support_count",
                "actual_signal_count",
                "actual_positive_signal_count",
                "actual_negative_signal_count",
                "audit_status",
                "issue_flags",
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\n=== ARTIST COVERAGE ===\n"
    )

    print(
        coverage.sort_values(
            "artist_name_fa"
        )
        .to_string(index=False)
    )

    print(
        "\nArtists without recommendations:"
    )

    missing_artists = coverage[
        coverage["recommendation_count"] == 0
    ]

    if missing_artists.empty:
        print("None")
    else:
        print(
            missing_artists[
                [
                    "artist_name_fa",
                    "category",
                    "release_count",
                    "positive_release_count",
                    "coverage_status",
                ]
            ]
            .to_string(index=False)
        )

    print("\nSaved quality audit:")
    print(OUTPUT_FILE)

    print("\nSaved artist coverage:")
    print(ARTIST_COVERAGE_FILE)


if __name__ == "__main__":
    main()
