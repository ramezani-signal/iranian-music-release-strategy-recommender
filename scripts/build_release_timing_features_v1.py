from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_feature_matrix_v2_final.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "release_strategy_features_v1_timing.csv"
)


LOCAL_TIMEZONE = "Asia/Tehran"


DAY_NAMES = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
    5: "saturday",
    6: "sunday",
}


def assign_daypart(hour):
    if 5 <= hour < 12:
        return "morning"

    if 12 <= hour < 17:
        return "afternoon"

    if 17 <= hour < 21:
        return "evening"

    return "night"


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

    date_source = None

    if "api_published_at" in df.columns:
        date_source = "api_published_at"

    elif "published_at" in df.columns:
        date_source = "published_at"

    else:
        raise ValueError(
            "No publication datetime column found."
        )

    df["release_datetime_utc"] = pd.to_datetime(
        df[date_source],
        utc=True,
        errors="coerce",
    )

    if df["release_datetime_utc"].isna().any():
        bad_rows = df.loc[
            df["release_datetime_utc"].isna(),
            [
                "record_id",
                "artist_name_fa",
                date_source,
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

    df["release_datetime_tehran"] = (
        df["release_datetime_utc"]
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    df["release_year"] = (
        df["release_datetime_tehran"].dt.year
    )

    df["release_month"] = (
        df["release_datetime_tehran"].dt.month
    )

    df["release_quarter"] = (
        df["release_datetime_tehran"].dt.quarter
    )

    df["release_day_of_month"] = (
        df["release_datetime_tehran"].dt.day
    )

    df["release_day_of_week_num"] = (
        df["release_datetime_tehran"].dt.dayofweek
    )

    df["release_day_of_week"] = (
        df["release_day_of_week_num"]
        .map(DAY_NAMES)
    )

    df["release_hour_tehran"] = (
        df["release_datetime_tehran"].dt.hour
    )

    df["release_minute_tehran"] = (
        df["release_datetime_tehran"].dt.minute
    )

    df["release_daypart"] = (
        df["release_hour_tehran"]
        .apply(assign_daypart)
    )

    # در تقویم کاری ایران، پنج‌شنبه و جمعه
    # به‌عنوان انتهای هفته در نظر گرفته می‌شوند.
    df["is_iran_weekend_release"] = (
        df["release_day_of_week_num"]
        .isin([3, 4])
        .astype(int)
    )

    # ترتیب زمانی انتشار در داخل هر هنرمند
    df = df.sort_values(
        [
            "artist_id",
            "release_datetime_utc",
            "video_id",
        ]
    ).copy()

    df["artist_release_sequence"] = (
        df.groupby("artist_id")
        .cumcount()
        + 1
    )

    df["previous_release_datetime_utc"] = (
        df.groupby("artist_id")[
            "release_datetime_utc"
        ]
        .shift(1)
    )

    df["days_since_previous_release"] = (
        (
            df["release_datetime_utc"]
            - df["previous_release_datetime_utc"]
        )
        .dt.total_seconds()
        / 86400
    )

    df["has_previous_release_in_dataset"] = (
        df["previous_release_datetime_utc"]
        .notna()
        .astype(int)
    )

    timing_numeric_columns = [
        "release_year",
        "release_month",
        "release_quarter",
        "release_day_of_month",
        "release_day_of_week_num",
        "release_hour_tehran",
        "release_minute_tehran",
        "is_iran_weekend_release",
        "artist_release_sequence",
        "has_previous_release_in_dataset",
    ]

    if df[timing_numeric_columns].isna().sum().sum() != 0:
        raise ValueError(
            "Missing values found in timing features."
        )

    if not np.isfinite(
        df[timing_numeric_columns]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite timing feature values detected."
        )

    invalid_gaps = df[
        df["days_since_previous_release"] < 0
    ]

    if not invalid_gaps.empty:
        print(
            "\nERROR: Negative release gaps:\n"
        )

        print(
            invalid_gaps[
                [
                    "record_id",
                    "artist_name_fa",
                    "release_datetime_utc",
                    "previous_release_datetime_utc",
                    "days_since_previous_release",
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            "Negative release interval detected."
        )

    df["strategy_timing_version"] = (
        "ST_TIMING_V1_TEHRAN"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== RELEASE TIMING FEATURES V1 ===\n"
    )

    print("Rows:", len(df))
    print("Timezone:", LOCAL_TIMEZONE)

    print(
        "\nPublication date range in Tehran time:"
    )

    print(
        "Earliest:",
        df["release_datetime_tehran"].min(),
    )

    print(
        "Latest:",
        df["release_datetime_tehran"].max(),
    )

    print(
        "\nRelease counts by weekday:\n"
    )

    weekday_order = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    print(
        df["release_day_of_week"]
        .value_counts()
        .reindex(
            weekday_order,
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nRelease counts by daypart:\n"
    )

    print(
        df["release_daypart"]
        .value_counts()
        .reindex(
            [
                "morning",
                "afternoon",
                "evening",
                "night",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nIran weekend release counts:\n"
    )

    print(
        df["is_iran_weekend_release"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nRelease hour distribution:\n"
    )

    print(
        df["release_hour_tehran"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nRelease timing by category:\n"
    )

    print(
        pd.crosstab(
            df["category"],
            df["release_daypart"],
        ).to_string()
    )

    print(
        "\nWeekend release by category:\n"
    )

    print(
        pd.crosstab(
            df["category"],
            df["is_iran_weekend_release"],
        ).to_string()
    )

    print(
        "\nRelease interval summary:\n"
    )

    print(
        df["days_since_previous_release"]
        .describe()
        .to_string()
    )

    print(
        "\nMedian release interval by artist:\n"
    )

    print(
        df.groupby("artist_name_fa")[
            "days_since_previous_release"
        ]
        .median()
        .sort_values()
        .round(2)
        .to_string()
    )

    print(
        "\nPerformance percentile by daypart:\n"
    )

    print(
        df.groupby("release_daypart")[
            "performance_percentile_final"
        ]
        .agg(
            [
                "count",
                "median",
                "mean",
            ]
        )
        .round(2)
        .to_string()
    )

    print(
        "\nPerformance percentile by weekday:\n"
    )

    weekday_performance = (
        df.groupby("release_day_of_week")[
            "performance_percentile_final"
        ]
        .agg(
            [
                "count",
                "median",
                "mean",
            ]
        )
        .reindex(weekday_order)
        .round(2)
    )

    print(
        weekday_performance.to_string()
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
