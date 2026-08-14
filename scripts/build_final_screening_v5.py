from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCREENING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v4_final.csv"
)

PROPOSAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_source_review_proposal.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v5_final.csv"
)


def main():
    screening = pd.read_csv(SCREENING_FILE)

    proposal = pd.read_csv(PROPOSAL_FILE)

    proposal_subset = proposal[
        [
            "video_id",
            "proposed_final_status",
            "proposed_final_reason",
            "final_review_confidence",
        ]
    ].copy()

    if proposal_subset["video_id"].duplicated().any():
        raise ValueError(
            "Proposal contains duplicate video IDs."
        )

    df = screening.merge(
        proposal_subset,
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    df["final_screening_status_v5"] = (
        df["final_screening_status_v4"]
        .astype("string")
    )

    df["final_screening_reason_v5"] = (
        df["final_screening_reason_v4"]
        .astype("string")
    )

    df["final_screening_decision_source_v5"] = (
        "final_screening_v4"
    )

    review_mask = (
        df["final_screening_status_v4"]
        == "final_needs_source_review"
    )

    missing_proposal_mask = (
        review_mask
        & df["proposed_final_status"].isna()
    )

    if missing_proposal_mask.any():
        missing_rows = df.loc[
            missing_proposal_mask,
            [
                "artist_name_fa",
                "video_id",
                "api_video_title",
            ],
        ]

        print(
            "\nERROR: Source-review rows without proposal:\n"
        )

        print(
            missing_rows.to_string(index=False)
        )

        raise ValueError(
            "Some source-review rows do not have "
            "a final proposal."
        )

    unexpected_proposal_mask = (
        ~review_mask
        & df["proposed_final_status"].notna()
    )

    if unexpected_proposal_mask.any():
        unexpected_rows = df.loc[
            unexpected_proposal_mask,
            [
                "artist_name_fa",
                "video_id",
                "final_screening_status_v4",
                "proposed_final_status",
            ],
        ]

        print(
            "\nERROR: Proposal applied to unexpected rows:\n"
        )

        print(
            unexpected_rows.to_string(index=False)
        )

        raise ValueError(
            "Proposal contains video IDs outside "
            "final_needs_source_review."
        )

    df.loc[
        review_mask,
        "final_screening_status_v5",
    ] = df.loc[
        review_mask,
        "proposed_final_status",
    ].values

    df.loc[
        review_mask,
        "final_screening_reason_v5",
    ] = df.loc[
        review_mask,
        "proposed_final_reason",
    ].values

    df.loc[
        review_mask,
        "final_screening_decision_source_v5",
    ] = (
        "manual_final_source_audit_2026-07-10"
    )

    df["final_review_confidence_v5"] = pd.NA

    df.loc[
        review_mask,
        "final_review_confidence_v5",
    ] = df.loc[
        review_mask,
        "final_review_confidence",
    ].values

    unresolved_count = (
        df["final_screening_status_v5"]
        == "final_needs_source_review"
    ).sum()

    if unresolved_count != 0:
        raise ValueError(
            f"{unresolved_count} source-review rows "
            "remain unresolved."
        )

    if len(df) != len(screening):
        raise ValueError(
            "Row count changed during merge."
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

    print(
        "\n=== FINAL SCREENING V5 STATUS COUNTS ===\n"
    )

    print(
        df["final_screening_status_v5"]
        .value_counts()
        .to_string()
    )

    print(
        "\n=== FINAL SCREENING V5 BY ARTIST ===\n"
    )

    print(
        pd.crosstab(
            df["artist_name_fa"],
            df["final_screening_status_v5"],
        ).to_string()
    )

    print(
        "\n=== V4 TO V5 TRANSITION MATRIX ===\n"
    )

    print(
        pd.crosstab(
            df["final_screening_status_v4"],
            df["final_screening_status_v5"],
        ).to_string()
    )

    print(
        "\nDecision source counts:\n"
    )

    print(
        df["final_screening_decision_source_v5"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal needs source review remaining:",
        (
            df["final_screening_status_v5"]
            == "final_needs_source_review"
        ).sum(),
    )

    print(
        "\nTotal rows:",
        len(df),
    )

    print(
        "\nSaved final integrated screening file:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
