from __future__ import annotations

import hashlib
import secrets
import smtplib
import sqlite3
import time
from email.message import EmailMessage
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=2,
)


class AuthBackend:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reset_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )

    def ensure_user(self, username: str, email: str, password: str) -> None:
        password_hash = ph.hash(password)
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    email=excluded.email,
                    password_hash=excluded.password_hash,
                    failed_attempts=0,
                    locked_until=0,
                    updated_at=excluded.updated_at
                """,
                (username.lower(), email.lower(), password_hash, now),
            )

    def authenticate(self, username: str, password: str) -> bool:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE username=? OR email=?",
                (username.lower(), username.lower()),
            ).fetchone()
            if row is None:
                return False
            if float(row["locked_until"]) > now:
                return False

            try:
                valid = ph.verify(row["password_hash"], password)
            except VerifyMismatchError:
                valid = False

            if valid:
                conn.execute(
                    "UPDATE users SET failed_attempts=0, locked_until=0, updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                return True

            attempts = int(row["failed_attempts"]) + 1
            locked_until = now + 900 if attempts >= 5 else 0
            conn.execute(
                "UPDATE users SET failed_attempts=?, locked_until=?, updated_at=? WHERE id=?",
                (attempts, locked_until, now, row["id"]),
            )
            return False

    def create_reset_token(self, identity: str) -> str | None:
        identity = identity.lower()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, email FROM users WHERE username=? OR email=?",
                (identity, identity),
            ).fetchone()
            if row is None:
                return None

            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            conn.execute(
                "UPDATE reset_tokens SET consumed=1 WHERE user_id=?",
                (row["id"],),
            )
            conn.execute(
                "INSERT INTO reset_tokens(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, row["id"], time.time() + 900),
            )
            return token

    def reset_password(self, token: str, new_password: str) -> bool:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM reset_tokens WHERE token_hash=? AND consumed=0 AND expires_at>?",
                (token_hash, now),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE users SET password_hash=?, failed_attempts=0, locked_until=0, updated_at=? WHERE id=?",
                (ph.hash(new_password), now, row["user_id"]),
            )
            conn.execute(
                "UPDATE reset_tokens SET consumed=1 WHERE token_hash=?",
                (token_hash,),
            )
            return True

    def send_reset_email(self, recipient: str, token: str, reset_url: str) -> bool:
        import os

        host = os.getenv("SMTP_HOST", "").strip()
        port = int(os.getenv("SMTP_PORT", "587"))
        username = os.getenv("SMTP_USERNAME", "").strip()
        password = os.getenv("SMTP_PASSWORD", "")
        sender = os.getenv("SMTP_FROM", username).strip()
        if not host or not username or not password:
            return False

        message = EmailMessage()
        message["Subject"] = "Password reset - Local AI Shorts Factory"
        message["From"] = sender
        message["To"] = recipient
        message.set_content(
            "A password reset was requested for your account.\n\n"
            f"Reset link: {reset_url}?token={token}\n\n"
            "This link expires in 15 minutes and can only be used once.\n"
            "If you did not request this, ignore this email."
        )

        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        return True
