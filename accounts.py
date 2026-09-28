from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path


ACCOUNT_DB = Path(__file__).resolve().parent / ".app_data" / "accounts.sqlite3"
PASSWORD_ITERATIONS = 310_000
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,32}\Z")


def initialize_accounts(database: Path = ACCOUNT_DB) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS accounts (
                    username TEXT PRIMARY KEY COLLATE NOCASE,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL
                )"""
            )


def register_account(username: str, password: str, database: Path = ACCOUNT_DB) -> None:
    username = username.strip()
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("Use 3-32 letters, numbers, dots, dashes, or underscores for the username.")
    if len(password) < 8 or len(password) > 256:
        raise ValueError("Choose a password between 8 and 256 characters.")

    initialize_accounts(database)
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    try:
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO accounts (username, salt, password_hash) VALUES (?, ?, ?)",
                    (username, salt, password_hash),
                )
    except sqlite3.IntegrityError as error:
        raise ValueError("That username is already registered.") from error


def authenticate_account(username: str, password: str, database: Path = ACCOUNT_DB) -> bool:
    if not database.is_file():
        return False
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT salt, password_hash FROM accounts WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return False

    salt, expected_hash = row
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return hmac.compare_digest(actual_hash, expected_hash)