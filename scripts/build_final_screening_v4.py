from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_V3_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v3.csv"
)

CONTENT_REVIEW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trusted_content_review_v2_final.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v4_final.csv"
)


def map_final_status(row):
    v3_status = row["screening_status_v3"]

    if v3_status == "likely_release":
        return (
            "final_likely_release",
            "accepted_from_v3_likely_release",
        )

    if v3_status == "likely_non_release":
        return (
            "final_non_release",
            "excluded_source_or_non_release_signal",
        )

    if v3_status == "needs_source_review":
        return (
            "final_needs_source_review",
            "source_registry_pending_review",
        )

    if v3_status == "trusted_non_release_signal":
        return (
            "final_non_release",
            "trusted_source_but_explicit_non_release_content_signal",
        )

    if v3_status == "trusted_needs_content_review":
        content_status = row.get(
            "content_review_status_v2"
        )

        if content_status == "likely_release":
            return (
                "final_likely_release",
                "accepted_after_trusted_content_review",
            )

        if content_status == "non_release_performance":
            return (
                "final_non_release_performance",
                "excluded_live_or_unplugged_performance",
            )

        if content_status == "non_release_modified_content":
            return (
                "final_non_release_modified_content",
                "excluded_modified_or_cross_era_content",
            )

        return (
            "final_unresolved",
            "missing_or_unknown_content_review_result",
        )

    return (
        "final_unresolved",
        "unmapped_v3_status",
    )


def main():
    screening = pd.read_csv(
        SCREENING_V3_FILE
    )

    content_review = pd.read_csv(
        CONTENT_REVIEW_FILE
    )

    review_subset = content_review[
        [
            "video_id",
            "content_review_status_v2",
            "content_review_reason_v2",
            "content_review_decision_source",
        ]
    ].copy()

    df = screening.merge(
        review_subset,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    expected_review_mask = (
        df["screening_status_v3"]
        == "trusted_needs_content_review"
    )

    missing_reviews = df[
        expected_review_mask
        & df["content_review_status_v2"].isna()
    ]

    if not missing_reviews.empty:
        print(
            "\nERROR: Missing content review decisions:\n"
        )

        print(
            missing_reviews[
                [
                    "video_id",
                    "artist_name_fa",
                    "api_video_title",
                    "screening_status_v3",
                ]
            ].to_string(index=False)
        )

        raise SystemExit(1)

    decisions = df.apply(
        map_final_status,
        axis=1,
        result_type="expand",
    )

    decisions.columns = [
        "final_screening_status_v4",
        "final_screening_reason_v4",
    ]

    df[
        [
            "final_screening_status_v4",
            "final_screening_reason_v4",
        ]
    ] = decisions

    unresolved = df[
        df["final_screening_status_v4"]
        == "final_unresolved"
    ]

    if not unresolved.empty:
        print(
            "\nERROR: Unresolved final decisions:\n"
        )

        print(
            unresolved[
                [
                    "video_id",
                    "artist_name_fa",
                    "api_video_title",
                    "screening_status_v3",
                    "content_review_status_v2",
                ]
            ].to_string(index=False)
        )

        raise SystemExit(1)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== FINAL SCREENING V4 STATUS COUNTS ===\n"
    )

    print(
        df["final_screening_status_v4"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== FINAL SCREENING V4 BY ARTIST ===\n"
    )

    print(
        pd.crosstab(
            df["artist_name_fa"],
            df["final_screening_status_v4"],
        ).to_string()
    )

    print(
        "\n=== V3 TO FINAL V4 TRANSITION MATRIX ===\n"
    )

    print(
        pd.crosstab(
            df["screening_status_v3"],
            df["final_screening_status_v4"],
        ).to_string()
    )

    print(
        "\nFinal unresolved rows:",
        (
            df["final_screening_status_v4"]
            == "final_unresolved"
        ).sum()
    )

    print(
        "\nTotal rows:",
        len(df)
    )

    print(
        "\nSaved final screening file:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
