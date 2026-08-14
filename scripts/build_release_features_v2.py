from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v2_features.csv"
)

SNAPSHOT_DATETIME = pd.Timestamp(
    "2026-07-10T23:59:59Z"
)


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    if df["record_id"].duplicated().any():
        raise ValueError(
            "Duplicate record_id detected."
        )

    df["published_datetime"] = pd.to_datetime(
        df["api_published_at"],
        utc=True,
        errors="coerce",
    )

    if df["published_datetime"].isna().any():
        bad_rows = df.loc[
            df["published_datetime"].isna(),
            [
                "record_id",
                "artist_name_fa",
                "api_published_at",
            ],
        ]

        print(
            "\nERROR: Invalid publication dates:\n"
        )
        print(
            bad_rows.to_string(index=False)
        )

        raise ValueError(
            "Invalid publication datetime detected."
        )

    df["snapshot_datetime"] = SNAPSHOT_DATETIME

    age_seconds = (
        df["snapshot_datetime"]
        - df["published_datetime"]
    ).dt.total_seconds()

    df["video_age_days"] = (
        age_seconds / 86400
    )

    if (df["video_age_days"] <= 0).any():
        bad_rows = df.loc[
            df["video_age_days"] <= 0,
            [
                "record_id",
                "artist_name_fa",
                "published_datetime",
                "video_age_days",
            ],
        ]

        print(
            "\nERROR: Non-positive video ages:\n"
        )
        print(
            bad_rows.to_string(index=False)
        )

        raise ValueError(
            "Non-positive video age detected."
        )

    df["video_age_years"] = (
        df["video_age_days"] / 365.25
    )

    df["views_per_day"] = (
        df["view_count"]
        / df["video_age_days"]
    )

    df["likes_per_1000_views"] = (
        df["like_count"]
        / df["view_count"]
        * 1000
    )

    df["comments_per_1000_views"] = (
        df["comment_count"]
        / df["view_count"]
        * 1000
    )

    df["engagement_count"] = (
        df["like_count"]
        + df["comment_count"]
    )

    df["engagement_rate"] = (
        df["engagement_count"]
        / df["view_count"]
    )

    df["duration_minutes"] = (
        df["duration_seconds"] / 60
    )

    df["log_view_count"] = np.log1p(
        df["view_count"]
    )

    df["log_like_count"] = np.log1p(
        df["like_count"]
    )

    df["log_comment_count"] = np.log1p(
        df["comment_count"]
    )

    df["log_views_per_day"] = np.log1p(
        df["views_per_day"]
    )

    feature_columns = [
        "video_age_days",
        "video_age_years",
        "views_per_day",
        "likes_per_1000_views",
        "comments_per_1000_views",
        "engagement_count",
        "engagement_rate",
        "duration_minutes",
        "log_view_count",
        "log_like_count",
        "log_comment_count",
        "log_views_per_day",
    ]

    missing_features = (
        df[feature_columns]
        .isna()
        .sum()
    )

    if missing_features.sum() != 0:
        print(
            "\nERROR: Missing engineered features:\n"
        )
        print(
            missing_features.to_string()
        )

        raise ValueError(
            "Missing engineered feature values detected."
        )

    numeric_feature_values = (
        df[feature_columns]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        numeric_feature_values
    ).all():
        raise ValueError(
            "Infinite engineered feature values detected."
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== FINAL RELEASE DATASET V2 FEATURES ===\n"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print(
        "\nEngineered feature columns:\n"
    )

    for col in feature_columns:
        print(col)

    print(
        "\nMissing engineered feature values:\n"
    )

    print(
        df[feature_columns]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nEngineered feature summary:\n"
    )

    print(
        df[feature_columns]
        .describe()
        .transpose()
        .to_string()
    )

    print(
        "\nPublication date range:\n"
    )

    print(
        "Earliest:",
        df["published_datetime"].min(),
    )

    print(
        "Latest:",
        df["published_datetime"].max(),
    )

    print(
        "Snapshot:",
        SNAPSHOT_DATETIME,
    )

    print(
        "\nTop 15 videos by views_per_day:\n"
    )

    print(
        df.sort_values(
            "views_per_day",
            ascending=False,
        )[
            [
                "artist_name_fa",
                "api_video_title",
                "published_datetime",
                "view_count",
                "video_age_days",
                "views_per_day",
                "engagement_rate",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nFeature summary by category:\n"
    )

    print(
        df.groupby("category")[
            [
                "view_count",
                "views_per_day",
                "engagement_rate",
                "duration_minutes",
            ]
        ]
        .median()
        .to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
