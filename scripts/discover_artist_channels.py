import os
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIST_FILE = PROJECT_ROOT / "data" / "raw" / "artist_seed_list.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "artist_channel_candidates.csv"

MAX_RESULTS_PER_ARTIST = 5


def load_artists():
    artists = []
    with open(ARTIST_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("include_in_main_dataset", "").strip().lower() == "yes":
                artists.append(row)
    return artists


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError("YOUTUBE_API_KEY was not found in .env")

    youtube = build("youtube", "v3", developerKey=api_key)

    artists = load_artists()
    results = []

    for artist in artists:
        artist_id = artist["artist_id"]
        artist_name = artist["artist_name_fa"]
        category = artist["category"]

        print(f"Searching channel candidates for: {artist_name}")

        request = youtube.search().list(
            part="snippet",
            q=artist_name,
            type="channel",
            maxResults=MAX_RESULTS_PER_ARTIST,
            relevanceLanguage="fa",
        )

        response = request.execute()

        for rank, item in enumerate(response.get("items", []), start=1):
            snippet = item.get("snippet", {})
            channel_id = item.get("id", {}).get("channelId", "")

            results.append(
                {
                    "artist_id": artist_id,
                    "artist_name_fa": artist_name,
                    "category": category,
                    "candidate_rank": rank,
                    "channel_id": channel_id,
                    "channel_title": snippet.get("channelTitle", ""),
                    "channel_description": snippet.get("description", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "source_query": artist_name,
                    "manual_status": "pending_review",
                    "candidate_type": "unknown",
                    "notes": "",
                }
            )

        time.sleep(0.2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "artist_id",
            "artist_name_fa",
            "category",
            "candidate_rank",
            "channel_id",
            "channel_title",
            "channel_description",
            "published_at",
            "source_query",
            "manual_status",
            "candidate_type",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} candidate channels to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
