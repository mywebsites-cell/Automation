"""Multi-account YouTube management.

Connects several Google accounts (each with its own OAuth token), stores a
label + topic keywords per account, and routes stories to the account whose
keywords match the story title/topic.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from modules.utils import load_json, write_json
from modules.youtube_uploader import YouTubeUploader

logger = logging.getLogger(__name__)


@dataclass
class YouTubeAccount:
    name: str
    keywords: list[str]
    channel_title: str
    token_path: Path
    story_prompt: str = ""

    @property
    def connected(self) -> bool:
        return self.token_path.exists()


class YouTubeAccountManager:
    """Manages multiple YouTube accounts under cache/youtube/accounts/<label>/."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.client_secrets_path = cache_dir / "youtube" / "client_secret.json"
        self.accounts_dir = cache_dir / "youtube" / "accounts"
        self.accounts_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_token()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _slug(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "account"

    def _account_dir(self, name: str) -> Path:
        return self.accounts_dir / self._slug(name)

    def _migrate_legacy_token(self) -> None:
        """Promote the old single-account token.json into a 'Default' account."""
        legacy = self.cache_dir / "youtube" / "token.json"
        if legacy.exists() and not any(self.accounts_dir.iterdir()):
            default_dir = self._account_dir("Default")
            default_dir.mkdir(parents=True, exist_ok=True)
            (default_dir / "token.json").write_bytes(legacy.read_bytes())
            write_json(
                default_dir / "account.json",
                {"name": "Default", "keywords": [], "channel_title": ""},
            )

    # ── queries ──────────────────────────────────────────────────────────────
    def has_client_secret(self) -> bool:
        return self.client_secrets_path.exists()

    def list_accounts(self) -> list[YouTubeAccount]:
        accounts: list[YouTubeAccount] = []
        for d in sorted(self.accounts_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "account.json"
            meta = load_json(meta_path) if meta_path.exists() else {}
            accounts.append(
                YouTubeAccount(
                    name=str(meta.get("name", d.name)),
                    keywords=[k.strip() for k in meta.get("keywords", []) if k.strip()],
                    channel_title=str(meta.get("channel_title", "")),
                    token_path=d / "token.json",
                    story_prompt=str(meta.get("story_prompt", "")),
                )
            )
        return accounts

    def resolve_account(
        self,
        text: str,
        accounts: list[YouTubeAccount] | None = None,
    ) -> YouTubeAccount | None:
        """Pick the account whose keywords appear in `text`.

        Priority order:
        1. First account whose keyword(s) appear in the text.
        2. An account with no keywords (the explicit catch-all / default).
        3. The first account (fallback).
        """
        accounts = accounts if accounts is not None else self.list_accounts()
        if not accounts:
            return None
        text_l = (text or "").lower()
        for acc in accounts:
            if acc.keywords and any(k.lower() in text_l for k in acc.keywords):
                return acc
        for acc in accounts:
            if not acc.keywords:
                return acc
        return accounts[0]

    # ── actions ──────────────────────────────────────────────────────────────
    def get_uploader(self, account: YouTubeAccount) -> YouTubeUploader:
        return YouTubeUploader(self.client_secrets_path, account.token_path)

    def connect_account(self, name: str, keywords: list[str], story_prompt: str = "") -> YouTubeAccount:
        """Run the OAuth flow for a new account and store its token + metadata."""
        if not self.has_client_secret():
            raise RuntimeError("Upload your client_secret.json first.")
        label = name.strip() or "Account"
        account_dir = self._account_dir(label)
        account_dir.mkdir(parents=True, exist_ok=True)
        token_path = account_dir / "token.json"

        uploader = YouTubeUploader(self.client_secrets_path, token_path)
        uploader.authenticate()

        # Channel title lookup needs an extra scope; don't let it block
        # registration if it fails, so the account is still saved correctly.
        channel_title = ""
        try:
            channel_title = uploader.get_channel_title()
        except Exception as exc:  # noqa: BLE001 - registration should not fail here
            logger.warning("Could not fetch channel title for %s: %s", label, exc)

        cleaned_keywords = [k.strip() for k in keywords if k.strip()]
        write_json(
            account_dir / "account.json",
            {
                "name": label,
                "keywords": cleaned_keywords,
                "channel_title": channel_title,
                "story_prompt": story_prompt.strip(),
            },
        )
        return YouTubeAccount(
            name=label,
            keywords=cleaned_keywords,
            channel_title=channel_title,
            token_path=token_path,
            story_prompt=story_prompt.strip(),
        )

    def delete_account(self, name: str) -> None:
        account_dir = self._account_dir(name)
        if account_dir.exists():
            shutil.rmtree(account_dir)
