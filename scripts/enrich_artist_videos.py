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
    / "artist_video_candidates_enriched.csv"
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

    video_ids = (
        videos["video_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Unique candidate video IDs found: "
        f"{len(video_ids)}"
    )

    video_data = {}

    for start in range(
        0,
        len(video_ids),
        50,
    ):
        batch_ids = video_ids[
            start:start + 50
        ]

        request = youtube.videos().list(
            part=(
                "snippet,"
                "contentDetails,"
                "statistics,"
                "status"
            ),
            id=",".join(batch_ids),
            maxResults=50,
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get(
                "snippet",
                {},
            )

            content_details = item.get(
                "contentDetails",
                {},
            )

            statistics = item.get(
                "statistics",
                {},
            )

            status = item.get(
                "status",
                {},
            )

            video_data[item["id"]] = {
                "api_video_title": snippet.get(
                    "title",
                    "",
                ),
                "api_video_description": snippet.get(
                    "description",
                    "",
                ),
                "api_channel_id": snippet.get(
                    "channelId",
                    "",
                ),
                "api_channel_title": snippet.get(
                    "channelTitle",
                    "",
                ),
                "api_published_at": snippet.get(
                    "publishedAt",
                    "",
                ),
                "category_id": snippet.get(
                    "categoryId",
                    "",
                ),
                "default_language": snippet.get(
                    "defaultLanguage",
                    "",
                ),
                "default_audio_language": snippet.get(
                    "defaultAudioLanguage",
                    "",
                ),
                "duration": content_details.get(
                    "duration",
                    "",
                ),
                "definition": content_details.get(
                    "definition",
                    "",
                ),
                "caption": content_details.get(
                    "caption",
                    "",
                ),
                "licensed_content": content_details.get(
                    "licensedContent",
                    "",
                ),
                "view_count": statistics.get(
                    "viewCount",
                    "",
                ),
                "like_count": statistics.get(
                    "likeCount",
                    "",
                ),
                "comment_count": statistics.get(
                    "commentCount",
                    "",
                ),
                "privacy_status": status.get(
                    "privacyStatus",
                    "",
                ),
                "embeddable": status.get(
                    "embeddable",
                    "",
                ),
                "made_for_kids": status.get(
                    "madeForKids",
                    "",
                ),
            }

    enriched_rows = []

    for _, row in videos.iterrows():
        row_data = row.to_dict()

        extra = video_data.get(
            str(row["video_id"]),
            {},
        )

        row_data.update(extra)

        enriched_rows.append(row_data)

    enriched_df = pd.DataFrame(
        enriched_rows
    )

    enriched_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Videos returned by API: "
        f"{len(video_data)}"
    )

    print(
        "Saved enriched candidate video data to:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
