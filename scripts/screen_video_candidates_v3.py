from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v2.csv"
)

REGISTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "source_registry_v5_all91_reviewed.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "artist_video_candidates_screened_v3.csv"
)


def classify_v3(row):
    v2_status = row["screening_status_v2"]
    source_status = row["source_status"]

    if source_status == "excluded":
        return (
            "likely_non_release",
            "excluded_source_registry"
        )

    if source_status == "pending_review":
        return (
            "needs_source_review",
            "pending_source_registry"
        )

    if source_status == "trusted":
        if v2_status == "likely_release":
            return (
                "likely_release",
                "v2_release_signal_and_trusted_source"
            )

        if v2_status == "needs_review":
            return (
                "trusted_needs_content_review",
                "trusted_source_but_v2_content_ambiguous"
            )

        if v2_status == "likely_non_release":
            return (
                "trusted_non_release_signal",
                "trusted_source_but_v2_non_release_signal"
            )

    return (
        "needs_review",
        "unresolved_v3_rule"
    )


def main():
    videos = pd.read_csv(VIDEO_FILE)

    registry = pd.read_csv(REGISTRY_FILE)

    source_info = registry[
        [
            "channel_id",
            "source_type",
            "source_status",
            "source_notes",
        ]
    ].copy()

    df = videos.merge(
        source_info,
        on="channel_id",
        how="left",
        validate="many_to_one",
    )

    if df["source_status"].isna().any():
        missing = df[
            df["source_status"].isna()
        ][
            [
                "video_id",
                "channel_id",
                "api_channel_title",
            ]
        ]

        print(
            "\nERROR: Videos with missing source registry status:\n"
        )
        print(missing.to_string(index=False))

        raise SystemExit(1)

    decisions = df.apply(
        classify_v3,
        axis=1,
        result_type="expand",
    )

    decisions.columns = [
        "screening_status_v3",
        "screening_reason_v3",
    ]

    df[
        [
            "screening_status_v3",
            "screening_reason_v3",
        ]
    ] = decisions

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== V3 SCREENING STATUS COUNTS ===\n")

    print(
        df["screening_status_v3"]
        .value_counts()
        .to_string()
    )

    print("\n=== V3 SCREENING BY ARTIST ===\n")

    print(
        pd.crosstab(
            df["artist_name_fa"],
            df["screening_status_v3"],
        ).to_string()
    )

    print(
        "\n=== V2 TO V3 TRANSITION MATRIX ===\n"
    )

    print(
        pd.crosstab(
            df["screening_status_v2"],
            df["screening_status_v3"],
        ).to_string()
    )

    print(
        "\n=== SOURCE STATUS TO V3 MATRIX ===\n"
    )

    print(
        pd.crosstab(
            df["source_status"],
            df["screening_status_v3"],
        ).to_string()
    )

    print("\nSaved V3 screened candidates to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
