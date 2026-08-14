from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_release_dataset_v3_performance_index.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "performance_index_sensitivity_v1.csv"
)


VIEW = "pi_view_component_clipped"
ENGAGEMENT = "pi_engagement_component_clipped"
COMMENT = "pi_comment_component_clipped"


WEIGHT_SCHEMES = {
    "A_60_25_15": {
        "view": 0.60,
        "engagement": 0.25,
        "comment": 0.15,
    },
    "B_70_20_10": {
        "view": 0.70,
        "engagement": 0.20,
        "comment": 0.10,
    },
    "C_65_25_10": {
        "view": 0.65,
        "engagement": 0.25,
        "comment": 0.10,
    },
}


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 115:
        raise ValueError(
            f"Expected 115 rows, found {len(df)}."
        )

    required_columns = [
        "record_id",
        "artist_name_fa",
        "category",
        "video_id",
        "api_video_title",
        VIEW,
        ENGAGEMENT,
        COMMENT,
        "log_views_per_day",
        "engagement_rate",
        "comments_per_1000_views",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        raw_col = f"pi_raw_{scheme_name}"
        pct_col = f"pi_pct_{scheme_name}"

        df[raw_col] = (
            weights["view"] * df[VIEW]
            + weights["engagement"] * df[ENGAGEMENT]
            + weights["comment"] * df[COMMENT]
        )

        df[pct_col] = (
            df.groupby("category")[raw_col]
            .rank(
                method="average",
                pct=True,
            )
            * 100
        ).round(2)

    percentile_columns = [
        f"pi_pct_{name}"
        for name in WEIGHT_SCHEMES
    ]

    raw_columns = [
        f"pi_raw_{name}"
        for name in WEIGHT_SCHEMES
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n=== PERFORMANCE INDEX "
        "SENSITIVITY ANALYSIS V1 ===\n"
    )

    print("Rows:", len(df))

    print(
        "\n=== PERCENTILE RANK "
        "SPEARMAN CORRELATION ===\n"
    )

    print(
        df[percentile_columns]
        .corr(method="spearman")
        .round(4)
        .to_string()
    )

    print(
        "\n=== RAW INDEX TO COMPONENT "
        "SPEARMAN CORRELATION ===\n"
    )

    component_columns = [
        "log_views_per_day",
        "engagement_rate",
        "comments_per_1000_views",
    ]

    correlation = (
        df[
            component_columns
            + raw_columns
        ]
        .corr(method="spearman")
        .loc[
            component_columns,
            raw_columns,
        ]
    )

    print(
        correlation
        .round(4)
        .to_string()
    )

    print(
        "\n=== TOP-10 OVERLAP ===\n"
    )

    top10_sets = {}

    for scheme_name in WEIGHT_SCHEMES:
        pct_col = f"pi_pct_{scheme_name}"
        raw_col = f"pi_raw_{scheme_name}"

        top10 = (
            df.sort_values(
                [pct_col, raw_col],
                ascending=False,
            )
            .head(10)
        )

        top10_sets[scheme_name] = set(
            top10["record_id"]
        )

        print(
            f"\n--- {scheme_name} ---\n"
        )

        print(
            top10[
                [
                    "artist_name_fa",
                    "category",
                    "api_video_title",
                    pct_col,
                    raw_col,
                ]
            ]
            .to_string(index=False)
        )

    scheme_names = list(
        WEIGHT_SCHEMES.keys()
    )

    print(
        "\n=== PAIRWISE TOP-10 OVERLAP ===\n"
    )

    for i in range(len(scheme_names)):
        for j in range(i + 1, len(scheme_names)):
            first = scheme_names[i]
            second = scheme_names[j]

            overlap = len(
                top10_sets[first]
                & top10_sets[second]
            )

            print(
                f"{first} vs {second}: "
                f"{overlap}/10"
            )

    print(
        "\n=== ABSOLUTE PERCENTILE "
        "DIFFERENCES ===\n"
    )

    pairs = [
        (
            "A_60_25_15",
            "B_70_20_10",
        ),
        (
            "A_60_25_15",
            "C_65_25_10",
        ),
        (
            "B_70_20_10",
            "C_65_25_10",
        ),
    ]

    for first, second in pairs:
        first_col = f"pi_pct_{first}"
        second_col = f"pi_pct_{second}"

        difference = (
            df[first_col]
            - df[second_col]
        ).abs()

        print(
            f"\n{first} vs {second}"
        )

        print(
            difference.describe()
            .round(3)
            .to_string()
        )

        print(
            "Rows with percentile "
            "difference > 10:",
            (difference > 10).sum(),
        )

        print(
            "Rows with percentile "
            "difference > 20:",
            (difference > 20).sum(),
        )

    print(
        "\n=== ARTIST MEDIAN PERCENTILES "
        "BY SCHEME ===\n"
    )

    artist_summary = (
        df.groupby(
            [
                "artist_name_fa",
                "category",
            ]
        )[percentile_columns]
        .median()
        .round(2)
        .reset_index()
    )

    print(
        artist_summary
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
