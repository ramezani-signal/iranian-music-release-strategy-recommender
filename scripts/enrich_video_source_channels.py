import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "artist_video_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "video_source_channels_enriched.csv"
)


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY was not found in .env"
        )

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key,
    )

    videos = pd.read_csv(INPUT_FILE)

    channel_summary = (
        videos
        .groupby(
            [
                "channel_id",
                "channel_title",
            ]
        )
        .agg(
            total_candidate_occurrences=(
                "video_id",
                "count",
            ),
            unique_artists_matched=(
                "artist_name_fa",
                "nunique",
            ),
        )
        .reset_index()
    )

    artist_matches = (
        videos
        .groupby("channel_id")["artist_name_fa"]
        .apply(
            lambda x: " | ".join(
                sorted(set(x.astype(str)))
            )
        )
        .reset_index(
            name="matched_artists"
        )
    )

    channel_summary = channel_summary.merge(
        artist_matches,
        on="channel_id",
        how="left",
    )

    channel_ids = (
        channel_summary["channel_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique video-source channels found: "
        f"{len(channel_ids)}"
    )

    channel_data = {}

    for start in range(
        0,
        len(channel_ids),
        50,
    ):
        batch_ids = channel_ids[
            start:start + 50
        ]

        request = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch_ids),
            maxResults=50,
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get(
                "snippet",
                {},
            )

            statistics = item.get(
                "statistics",
                {},
            )

            channel_data[item["id"]] = {
                "api_channel_title": (
                    snippet.get("title", "")
                ),
                "api_channel_description": (
                    snippet.get(
                        "description",
                        "",
                    )
                ),
                "custom_url": snippet.get(
                    "customUrl",
                    "",
                ),
                "country": snippet.get(
                    "country",
                    "",
                ),
                "channel_published_at": (
                    snippet.get(
                        "publishedAt",
                        "",
                    )
                ),
                "view_count": statistics.get(
                    "viewCount",
                    "",
                ),
                "subscriber_count": (
                    statistics.get(
                        "subscriberCount",
                        "",
                    )
                ),
                "hidden_subscriber_count": (
                    statistics.get(
                        "hiddenSubscriberCount",
                        "",
                    )
                ),
                "video_count": statistics.get(
                    "videoCount",
                    "",
                ),
            }

    enriched_rows = []

    for _, row in channel_summary.iterrows():
        row_data = row.to_dict()

        extra = channel_data.get(
            str(row["channel_id"]),
            {},
        )

        row_data.update(extra)

        enriched_rows.append(row_data)

    enriched_df = pd.DataFrame(
        enriched_rows
    )

    enriched_df = enriched_df.sort_values(
        [
            "total_candidate_occurrences",
            "unique_artists_matched",
        ],
        ascending=[
            False,
            False,
        ],
    )

    enriched_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Channels returned by API: "
        f"{len(channel_data)}"
    )

    print(
        "Saved enriched video-source "
        "channel data to:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
