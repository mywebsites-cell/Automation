"""YouTube Data API v3 uploader with resumable upload and OAuth2 token caching."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


@dataclass
class YouTubeUploadResult:
    video_id: str
    video_url: str


class YouTubeUploader:
    def __init__(self, client_secrets_path: Path, token_path: Path):
        self.client_secrets_path = client_secrets_path
        self.token_path = token_path
        self._service = None

    def authenticate(self):
        """Authenticate against YouTube (reusing a saved token when possible)."""
        if self._service is not None:
            return self._service

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if self.token_path.exists():
            # Load with the token's OWN scopes (do NOT pass SCOPES). Passing a
            # newer scope list here would make the refresh request include the
            # new scope, which Google rejects with "invalid_scope" for refresh
            # tokens that were granted a different scope set.
            creds = Credentials.from_authorized_user_file(str(self.token_path))

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.client_secrets_path.exists():
                    raise RuntimeError(
                        f"Missing OAuth client secret: {self.client_secrets_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.client_secrets_path), SCOPES
                )
                # Opens the user's browser on localhost for the OAuth consent screen.
                creds = flow.run_local_server(port=0)

            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        from googleapiclient.discovery import build

        self._service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)
        return self._service

    def upload_video(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        category_id: str,
        privacy_status: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> YouTubeUploadResult:
        """Upload a video file with a resumable upload, returning the video id/url."""
        service = self.authenticate()

        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path),
            chunksize=8 * 1024 * 1024,  # 8 MB chunks: far fewer round-trips to drop
            mimetype="video/*",
            resumable=True,
        )

        request = service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        attempt = 0
        max_attempts = 6
        while response is None:
            try:
                status, response = request.next_chunk(num_retries=3)
                if status:
                    pct = status.progress() * 100
                    logger.info("YouTube upload progress: %.1f%%", pct)
                    if progress_callback:
                        progress_callback(pct)
            except (ConnectionError, TimeoutError, OSError, HttpError) as exc:
                # The connection was dropped mid-upload (e.g. WinError 10054).
                # The MediaFileUpload object keeps its resumable session + byte
                # offset, so calling next_chunk() again resumes where it left off.
                attempt += 1
                if attempt >= max_attempts:
                    raise
                wait = min(2 ** attempt, 60)  # exponential backoff, capped at 60s
                logger.warning(
                    "Upload interrupted (%s); retrying in %ds (attempt %d/%d)",
                    exc, wait, attempt, max_attempts,
                )
                time.sleep(wait)

        video_id = response["id"]
        return YouTubeUploadResult(
            video_id=video_id,
            video_url=f"https://youtu.be/{video_id}",
        )

    def set_thumbnail(self, video_id: str, image_path: Path) -> None:
        """Set the custom thumbnail image for an uploaded video."""
        service = self.authenticate()
        from googleapiclient.http import MediaFileUpload

        service.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(image_path), mimetype="image/png"),
        ).execute()
        logger.info("Thumbnail set for video %s", video_id)

    def get_channel_title(self) -> str:
        """Return the title of the channel owned by the authenticated user."""
        service = self.authenticate()
        response = service.channels().list(part="snippet", mine=True).execute()
        items = response.get("items", [])
        if items:
            return items[0]["snippet"].get("title", "")
        return ""
