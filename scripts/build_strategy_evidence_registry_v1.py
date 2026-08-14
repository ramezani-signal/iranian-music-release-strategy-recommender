from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_evidence_registry_v1.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "strategy_evidence_registry_v1.json"
)


EVIDENCE_ROWS = [
    {
        "signal": "title_has_music_video",
        "feature_family": "title_format",
        "predictive_importance": "high",
        "global_direction": "negative",
        "category_adjusted_direction": "negative",
        "artist_adjusted_direction": "neutral",
        "artist_adjusted_supported": False,
        "recommendation_scope": "artist_specific_only",
        "recommendation_strength": "cautious",
        "allowed_in_engine": True,
        "rule_type": "contextual_marker",
        "evidence_grade": "B_context_dependent",
        "recommended_interpretation": (
            "Use only relative to the artist's own historical "
            "performance. Do not issue a universal negative rule."
        ),
    },
    {
        "signal": "title_has_lyric_video",
        "feature_family": "title_format",
        "predictive_importance": "moderate",
        "global_direction": "positive",
        "category_adjusted_direction": "positive",
        "artist_adjusted_direction": "weak_positive",
        "artist_adjusted_supported": False,
        "recommendation_scope": "artist_or_category_exploratory",
        "recommendation_strength": "exploratory",
        "allowed_in_engine": True,
        "rule_type": "exploratory_opportunity",
        "evidence_grade": "C_exploratory",
        "recommended_interpretation": (
            "Present as a testable option when similar releases "
            "performed well for the artist or category."
        ),
    },
    {
        "signal": "duration_minutes",
        "feature_family": "duration",
        "predictive_importance": "moderate",
        "global_direction": "positive",
        "category_adjusted_direction": "positive",
        "artist_adjusted_direction": "near_zero",
        "artist_adjusted_supported": False,
        "recommendation_scope": "artist_specific_only",
        "recommendation_strength": "cautious",
        "allowed_in_engine": True,
        "rule_type": "personalized_range",
        "evidence_grade": "B_context_dependent",
        "recommended_interpretation": (
            "Recommend duration ranges from the artist's own "
            "higher-performing releases, not a universal duration."
        ),
    },
    {
        "signal": "release_hour_tehran",
        "feature_family": "timing",
        "predictive_importance": "negative_or_unstable",
        "global_direction": "weak_positive",
        "category_adjusted_direction": "heterogeneous",
        "artist_adjusted_direction": "heterogeneous",
        "artist_adjusted_supported": False,
        "recommendation_scope": "descriptive_only",
        "recommendation_strength": "insufficient",
        "allowed_in_engine": False,
        "rule_type": "unsupported",
        "evidence_grade": "D_insufficient",
        "recommended_interpretation": (
            "Show historical timing patterns only; do not claim "
            "an optimal universal release hour."
        ),
    },
    {
        "signal": "is_iran_weekend_release",
        "feature_family": "timing",
        "predictive_importance": "negative_or_unstable",
        "global_direction": "mixed",
        "category_adjusted_direction": "mixed",
        "artist_adjusted_direction": "mixed",
        "artist_adjusted_supported": False,
        "recommendation_scope": "descriptive_only",
        "recommendation_strength": "insufficient",
        "allowed_in_engine": False,
        "rule_type": "unsupported",
        "evidence_grade": "D_insufficient",
        "recommended_interpretation": (
            "Do not recommend weekday/weekend timing as a causal rule."
        ),
    },
    {
        "signal": "days_since_previous_release",
        "feature_family": "release_cadence",
        "predictive_importance": "unstable",
        "global_direction": "weak_positive",
        "category_adjusted_direction": "heterogeneous",
        "artist_adjusted_direction": "artist_specific",
        "artist_adjusted_supported": False,
        "recommendation_scope": "artist_specific_only",
        "recommendation_strength": "exploratory",
        "allowed_in_engine": True,
        "rule_type": "personalized_cadence",
        "evidence_grade": "C_exploratory",
        "recommended_interpretation": (
            "Recommend only from the artist's own successful cadence "
            "distribution and display uncertainty."
        ),
    },
    {
        "signal": "title_character_count",
        "feature_family": "title_structure",
        "predictive_importance": "negative_or_unstable",
        "global_direction": "weak",
        "category_adjusted_direction": "heterogeneous",
        "artist_adjusted_direction": "heterogeneous",
        "artist_adjusted_supported": False,
        "recommendation_scope": "descriptive_only",
        "recommendation_strength": "insufficient",
        "allowed_in_engine": False,
        "rule_type": "unsupported",
        "evidence_grade": "D_insufficient",
        "recommended_interpretation": (
            "Retain for descriptive reporting but exclude from "
            "strong recommendation generation."
        ),
    },
    {
        "signal": "title_word_count",
        "feature_family": "title_structure",
        "predictive_importance": "negative_or_unstable",
        "global_direction": "weak",
        "category_adjusted_direction": "heterogeneous",
        "artist_adjusted_direction": "heterogeneous",
        "artist_adjusted_supported": False,
        "recommendation_scope": "descriptive_only",
        "recommendation_strength": "insufficient",
        "allowed_in_engine": False,
        "rule_type": "unsupported",
        "evidence_grade": "D_insufficient",
        "recommended_interpretation": (
            "Retain for descriptive reporting but exclude from "
            "strong recommendation generation."
        ),
    },
]


def main():
    registry = pd.DataFrame(EVIDENCE_ROWS)

    required_columns = [
        "signal",
        "feature_family",
        "predictive_importance",
        "global_direction",
        "category_adjusted_direction",
        "artist_adjusted_direction",
        "recommendation_scope",
        "recommendation_strength",
        "allowed_in_engine",
        "rule_type",
        "evidence_grade",
        "recommended_interpretation",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in registry.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing registry columns: {missing_columns}"
        )

    if registry["signal"].duplicated().any():
        raise ValueError(
            "Duplicate strategy signals detected."
        )

    registry["evidence_registry_version"] = (
        "STRATEGY_EVIDENCE_V1"
    )

    registry.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "registry_version":
            "STRATEGY_EVIDENCE_V1",
        "methodological_principles": [
            "Observational associations are not causal effects.",
            "Artist-adjusted evidence has priority over global association.",
            "Unsupported signals must not generate strong recommendations.",
            "Context-dependent signals require artist or category history.",
            "Recommendations must include confidence and evidence grade.",
        ],
        "evidence_grades": {
            "A_robust":
                "Stable after artist and category adjustment.",
            "B_context_dependent":
                "Predictive or descriptive signal dependent on context.",
            "C_exploratory":
                "Promising but limited or statistically uncertain evidence.",
            "D_insufficient":
                "Unstable, sparse, or unsupported for recommendation.",
        },
        "signals":
            registry.to_dict(
                orient="records"
            ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== STRATEGY EVIDENCE REGISTRY V1 ===\n"
    )

    print("Signals:", len(registry))

    print(
        "\nEvidence-grade counts:\n"
    )

    print(
        registry["evidence_grade"]
        .value_counts()
        .to_string()
    )

    print(
        "\nAllowed in recommendation engine:\n"
    )

    print(
        registry["allowed_in_engine"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nRecommendation scope counts:\n"
    )

    print(
        registry["recommendation_scope"]
        .value_counts()
        .to_string()
    )

    print(
        "\nComplete evidence registry:\n"
    )

    print(
        registry[
            [
                "signal",
                "feature_family",
                "predictive_importance",
                "global_direction",
                "artist_adjusted_direction",
                "recommendation_scope",
                "recommendation_strength",
                "allowed_in_engine",
                "evidence_grade",
            ]
        ]
        .to_string(index=False)
    )

    print(
        "\nSignals allowed in engine:\n"
    )

    allowed = registry[
        registry["allowed_in_engine"]
        == True
    ]

    print(
        allowed[
            [
                "signal",
                "rule_type",
                "recommendation_scope",
                "recommendation_strength",
                "evidence_grade",
            ]
        ]
        .to_string(index=False)
    )

    print("\nSaved evidence CSV:")
    print(OUTPUT_CSV)

    print("\nSaved evidence JSON:")
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
