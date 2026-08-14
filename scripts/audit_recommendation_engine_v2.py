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
    / "artist_strategy_recommendations_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_engine_v2_quality_audit.csv"
)

ARTIST_AUDIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendation_engine_v2_artist_audit.csv"
)


TARGET = "is_high_or_top"

EXPECTED_ENGINE_VERSION = "RE_V2"

EXPECTED_SIGNALS = {
    "title_has_music_video",
    "title_has_lyric_video",
    "duration_minutes",
    "days_since_previous_release",
    "insufficient_artist_evidence",
}

VALID_STATUSES = {
    "active_recommendation",
    "category_fallback_recommendation",
    "historical_observation",
    "insufficient_artist_evidence",
}

VALID_CONFIDENCE = {
    "insufficient",
    "low",
    "low_medium",
    "medium",
    "medium_high",
}


def add_issue(issues, issue):
    if issue not in issues:
        issues.append(issue)


def audit_row(
    row,
    data,
):
    artist = row["artist_name_fa"]
    category = row["category"]
    signal = row["signal"]
    status = row["recommendation_status"]
    scope = row["recommendation_scope"]

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

    category_positive = category_df[
        category_df[TARGET] == 1
    ]

    category_negative = category_df[
        category_df[TARGET] == 0
    ]

    issues = []

    actual_support_count = None
    actual_signal_total = 0
    actual_positive_signal = 0
    actual_negative_signal = 0
    actual_reference_positive = 0
    actual_reference_negative = 0
    actual_reference_total = 0

    if signal == "insufficient_artist_evidence":
        actual_support_count = int(
            artist_df[TARGET].sum()
        )

        actual_reference_positive = (
            actual_support_count
        )

        actual_reference_negative = (
            len(artist_df)
            - actual_support_count
        )

        actual_reference_total = len(
            artist_df
        )

        if actual_support_count >= 2:
            add_issue(
                issues,
                "insufficient_status_but_artist_has_2_or_more_positives",
            )

        if status != "insufficient_artist_evidence":
            add_issue(
                issues,
                "insufficient_signal_wrong_status",
            )

        if row["confidence"] != "insufficient":
            add_issue(
                issues,
                "insufficient_signal_wrong_confidence",
            )

    elif signal in {
        "title_has_music_video",
        "title_has_lyric_video",
    }:
        if scope.startswith("category_"):
            reference_df = category_df
            positive_df = category_positive
            negative_df = category_negative
        else:
            reference_df = artist_df
            positive_df = artist_positive
            negative_df = artist_negative

        actual_positive_signal = int(
            positive_df[signal].sum()
        )

        actual_negative_signal = int(
            negative_df[signal].sum()
        )

        actual_signal_total = (
            actual_positive_signal
            + actual_negative_signal
        )

        actual_reference_positive = len(
            positive_df
        )

        actual_reference_negative = len(
            negative_df
        )

        actual_reference_total = len(
            reference_df
        )

        if signal == "title_has_music_video":
            actual_support_count = (
                actual_signal_total
            )

            if scope != "artist_specific":
                add_issue(
                    issues,
                    "music_video_not_artist_specific",
                )

            if actual_signal_total < 2:
                add_issue(
                    issues,
                    "music_video_total_support_below_2",
                )

        else:
            if scope.startswith("category_"):
                actual_support_count = (
                    actual_positive_signal
                )

                if actual_positive_signal < 2:
                    add_issue(
                        issues,
                        "category_lyric_positive_support_below_2",
                    )

                if status != (
                    "category_fallback_recommendation"
                ):
                    add_issue(
                        issues,
                        "category_lyric_wrong_status",
                    )
            else:
                actual_support_count = (
                    actual_positive_signal
                )

                if actual_positive_signal < 1:
                    add_issue(
                        issues,
                        "artist_lyric_positive_support_below_1",
                    )

    elif signal == "duration_minutes":
        actual_support_count = len(
            artist_positive[
                artist_positive[
                    "duration_minutes"
                ].notna()
            ]
        )

        actual_reference_positive = (
            actual_support_count
        )

        actual_reference_negative = len(
            artist_negative[
                artist_negative[
                    "duration_minutes"
                ].notna()
            ]
        )

        actual_reference_total = (
            actual_reference_positive
            + actual_reference_negative
        )

        if actual_support_count == 2:
            if status != "historical_observation":
                add_issue(
                    issues,
                    "two_duration_observations_not_historical",
                )

            if row["confidence"] != "low":
                add_issue(
                    issues,
                    "two_duration_observations_confidence_not_low",
                )

        if actual_support_count >= 3:
            if status != "active_recommendation":
                add_issue(
                    issues,
                    "duration_support_3_plus_not_active",
                )

    elif signal == "days_since_previous_release":
        actual_support_count = len(
            artist_positive[
                artist_positive[
                    "days_since_previous_release"
                ].notna()
            ]
        )

        actual_reference_positive = (
            actual_support_count
        )

        actual_reference_negative = len(
            artist_negative[
                artist_negative[
                    "days_since_previous_release"
                ].notna()
            ]
        )

        # Semantic contract:
        # reference_total_count represents all releases
        # available for the artist, while support_count
        # represents positive releases with a valid
        # previous-release gap.
        actual_reference_total = len(
            artist_df
        )

        if actual_support_count == 2:
            if status != "historical_observation":
                add_issue(
                    issues,
                    "two_cadence_observations_not_historical",
                )

            if row["confidence"] != "low":
                add_issue(
                    issues,
                    "two_cadence_observations_confidence_not_low",
                )

        if actual_support_count >= 3:
            if status != "active_recommendation":
                add_issue(
                    issues,
                    "cadence_support_3_plus_not_active",
                )

    else:
        add_issue(
            issues,
            "unexpected_signal",
        )

        actual_support_count = -1

    if int(row["support_count"]) != int(
        actual_support_count
    ):
        add_issue(
            issues,
            "support_count_mismatch",
        )

    if int(row["signal_total_count"]) != int(
        actual_signal_total
    ):
        add_issue(
            issues,
            "signal_total_count_mismatch",
        )

    if int(row["positive_signal_count"]) != int(
        actual_positive_signal
    ):
        add_issue(
            issues,
            "positive_signal_count_mismatch",
        )

    if int(row["negative_signal_count"]) != int(
        actual_negative_signal
    ):
        add_issue(
            issues,
            "negative_signal_count_mismatch",
        )

    if int(
        row["reference_positive_count"]
    ) != int(actual_reference_positive):
        add_issue(
            issues,
            "reference_positive_count_mismatch",
        )

    if int(
        row["reference_total_count"]
    ) != int(actual_reference_total):
        add_issue(
            issues,
            "reference_total_count_mismatch",
        )

    if status not in VALID_STATUSES:
        add_issue(
            issues,
            "invalid_recommendation_status",
        )

    if row["confidence"] not in VALID_CONFIDENCE:
        add_issue(
            issues,
            "invalid_confidence",
        )

    if (
        status == "historical_observation"
        and int(row["support_count"]) != 2
    ):
        add_issue(
            issues,
            "historical_observation_support_not_2",
        )

    if (
        status == "active_recommendation"
        and int(row["support_count"]) < 2
    ):
        add_issue(
            issues,
            "active_recommendation_support_below_2",
        )

    if (
        row["confidence"] == "medium_high"
        and int(row["support_count"]) < 8
    ):
        add_issue(
            issues,
            "medium_high_support_below_8",
        )

    if (
        row["confidence"] == "medium"
        and int(row["support_count"]) < 4
    ):
        add_issue(
            issues,
            "medium_support_below_4",
        )

    return {
        "artist_name_fa":
            artist,

        "category":
            category,

        "signal":
            signal,

        "recommendation_status":
            status,

        "recommendation_scope":
            scope,

        "reported_confidence":
            row["confidence"],

        "reported_support_count":
            int(row["support_count"]),

        "actual_support_count":
            int(actual_support_count),

        "reported_signal_total_count":
            int(row["signal_total_count"]),

        "actual_signal_total_count":
            int(actual_signal_total),

        "reported_positive_signal_count":
            int(row["positive_signal_count"]),

        "actual_positive_signal_count":
            int(actual_positive_signal),

        "reported_negative_signal_count":
            int(row["negative_signal_count"]),

        "actual_negative_signal_count":
            int(actual_negative_signal),

        "reported_reference_positive_count":
            int(row["reference_positive_count"]),

        "actual_reference_positive_count":
            int(actual_reference_positive),

        "reported_reference_total_count":
            int(row["reference_total_count"]),

        "actual_reference_total_count":
            int(actual_reference_total),

        "issue_count":
            len(issues),

        "issue_flags":
            " | ".join(issues),

        "audit_status":
            (
                "pass"
                if len(issues) == 0
                else "needs_revision"
            ),
    }


def main():
    data = pd.read_csv(DATA_FILE)

    recommendations = pd.read_csv(
        RECOMMENDATION_FILE
    )

    if len(data) != 115:
        raise ValueError(
            f"Expected 115 data rows, "
            f"found {len(data)}."
        )

    if data["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    if recommendations.empty:
        raise ValueError(
            "Recommendation V2 file is empty."
        )

    engine_versions = set(
        recommendations[
            "recommendation_engine_version"
        ].dropna()
    )

    if engine_versions != {
        EXPECTED_ENGINE_VERSION
    }:
        raise ValueError(
            "Unexpected engine version: "
            f"{engine_versions}"
        )

    unexpected_signals = (
        set(recommendations["signal"])
        - EXPECTED_SIGNALS
    )

    if unexpected_signals:
        raise ValueError(
            "Unexpected signals: "
            f"{sorted(unexpected_signals)}"
        )

    duplicate_key_count = (
        recommendations.duplicated(
            subset=[
                "artist_name_fa",
                "signal",
                "recommendation_status",
                "recommendation_scope",
            ]
        )
        .sum()
    )

    audit_rows = [
        audit_row(
            row,
            data,
        )
        for _, row
        in recommendations.iterrows()
    ]

    audit = pd.DataFrame(
        audit_rows
    )

    artist_base = (
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

    output_summary = (
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
            insufficient_row_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "insufficient_artist_evidence"
                    ).sum()
                ),
            ),
            active_row_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "active_recommendation"
                    ).sum()
                ),
            ),
            fallback_row_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "category_fallback_recommendation"
                    ).sum()
                ),
            ),
            historical_row_count=(
                "recommendation_status",
                lambda values: int(
                    (
                        values
                        == "historical_observation"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    artist_audit = artist_base.merge(
        output_summary,
        on=[
            "artist_name_fa",
            "category",
        ],
        how="left",
    )

    count_columns = [
        "output_row_count",
        "insufficient_row_count",
        "active_row_count",
        "fallback_row_count",
        "historical_row_count",
    ]

    artist_audit[count_columns] = (
        artist_audit[count_columns]
        .fillna(0)
        .astype(int)
    )

    artist_issues = []

    for _, row in artist_audit.iterrows():
        issues = []

        positive_count = int(
            row["positive_release_count"]
        )

        insufficient_count = int(
            row["insufficient_row_count"]
        )

        if positive_count < 2:
            if insufficient_count != 1:
                add_issue(
                    issues,
                    "artist_below_2_positives_missing_single_insufficient_row",
                )
        else:
            if insufficient_count != 0:
                add_issue(
                    issues,
                    "artist_2_plus_positives_has_insufficient_row",
                )

        if int(row["output_row_count"]) == 0:
            add_issue(
                issues,
                "artist_missing_from_output",
            )

        artist_issues.append(
            {
                **row.to_dict(),

                "issue_count":
                    len(issues),

                "issue_flags":
                    " | ".join(issues),

                "audit_status":
                    (
                        "pass"
                        if len(issues) == 0
                        else "needs_revision"
                    ),
            }
        )

    artist_audit = pd.DataFrame(
        artist_issues
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    artist_audit.to_csv(
        ARTIST_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RECOMMENDATION ENGINE V2 "
        "QUALITY AUDIT ===\n"
    )

    print(
        "Data rows:",
        len(data),
    )

    print(
        "Recommendation rows:",
        len(recommendations),
    )

    print(
        "Input artists:",
        data["artist_name_fa"].nunique(),
    )

    print(
        "Output artists:",
        recommendations[
            "artist_name_fa"
        ].nunique(),
    )

    print(
        "Duplicate recommendation keys:",
        duplicate_key_count,
    )

    print(
        "\nRow-level audit status counts:\n"
    )

    print(
        audit["audit_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nArtist-level audit status counts:\n"
    )

    print(
        artist_audit["audit_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nIssue-flag counts:\n"
    )

    issue_series = (
        audit["issue_flags"]
        .str.split(r" \| ")
        .explode()
    )

    issue_series = issue_series[
        issue_series != ""
    ]

    if issue_series.empty:
        print("None")
    else:
        print(
            issue_series
            .value_counts()
            .to_string()
        )

    print(
        "\n=== COMPLETE ROW-LEVEL AUDIT ===\n"
    )

    print(
        audit[
            [
                "artist_name_fa",
                "signal",
                "recommendation_status",
                "reported_confidence",
                "reported_support_count",
                "actual_support_count",
                "reported_signal_total_count",
                "actual_signal_total_count",
                "reported_positive_signal_count",
                "actual_positive_signal_count",
                "reported_negative_signal_count",
                "actual_negative_signal_count",
                "reported_reference_positive_count",
                "actual_reference_positive_count",
                "reported_reference_total_count",
                "actual_reference_total_count",
                "audit_status",
                "issue_flags",
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\n=== COMPLETE ARTIST-LEVEL AUDIT ===\n"
    )

    print(
        artist_audit[
            [
                "artist_name_fa",
                "category",
                "release_count",
                "positive_release_count",
                "output_row_count",
                "insufficient_row_count",
                "active_row_count",
                "fallback_row_count",
                "historical_row_count",
                "audit_status",
                "issue_flags",
            ]
        ]
        .sort_values(
            "artist_name_fa"
        )
        .to_string(index=False)
    )

    print(
        "\nFinal unresolved row issues:",
        (
            audit["audit_status"]
            == "needs_revision"
        ).sum(),
    )

    print(
        "Final unresolved artist issues:",
        (
            artist_audit["audit_status"]
            == "needs_revision"
        ).sum(),
    )

    print("\nSaved row-level audit:")
    print(OUTPUT_FILE)

    print("\nSaved artist-level audit:")
    print(ARTIST_AUDIT_FILE)


if __name__ == "__main__":
    main()
