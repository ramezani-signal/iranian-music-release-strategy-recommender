import os

from dotenv import load_dotenv
from googleapiclient.discovery import build


load_dotenv()

api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    raise ValueError("YOUTUBE_API_KEY was not found in the .env file.")

youtube = build(
    "youtube",
    "v3",
    developerKey=api_key,
)

request = youtube.videos().list(
    part="snippet,statistics",
    id="dQw4w9WgXcQ",
)

response = request.execute()

if not response.get("items"):
    raise ValueError("No video data was returned by the YouTube API.")

video = response["items"][0]

print("YouTube Data API connection: SUCCESS")
print("Video title:", video["snippet"]["title"])
print("Channel title:", video["snippet"]["channelTitle"])
print("View count:", video["statistics"].get("viewCount", "N/A"))

