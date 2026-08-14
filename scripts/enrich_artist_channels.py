import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "artist_channel_candidates.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "artist_channel_candidates_enriched.csv"


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError("YOUTUBE_API_KEY was not found in .env")

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key,
    )

    df = pd.read_csv(INPUT_FILE)

    channel_ids = (
        df["channel_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"Unique channel IDs found: {len(channel_ids)}")

    channel_data = {}

    for start in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[start:start + 50]

        request = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(batch_ids),
            maxResults=50,
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            channel_data[item["id"]] = {
                "api_channel_title": snippet.get("title", ""),
                "api_channel_description": snippet.get("description", ""),
                "custom_url": snippet.get("customUrl", ""),
                "country": snippet.get("country", ""),
                "channel_published_at": snippet.get("publishedAt", ""),
                "view_count": statistics.get("viewCount", ""),
                "subscriber_count": statistics.get("subscriberCount", ""),
                "hidden_subscriber_count": statistics.get(
                    "hiddenSubscriberCount",
                    ""
                ),
                "video_count": statistics.get("videoCount", ""),
            }

    enriched_rows = []

    for _, row in df.iterrows():
        row_data = row.to_dict()

        extra = channel_data.get(
            str(row["channel_id"]),
            {}
        )

        row_data.update(extra)
        enriched_rows.append(row_data)

    enriched_df = pd.DataFrame(enriched_rows)

    enriched_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Channels returned by API: {len(channel_data)}")
    print(f"Saved enriched data to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
