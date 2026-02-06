"""
SQLite storage for authentication data (user credentials + JWT secret).
"""
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = Path(__file__).parent / "auth.db"


def get_connection() -> sqlite3.Connection:
    """Get a connection to the auth database, creating tables if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            jwt_secret TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def is_initialized() -> bool:
    """Check if auth has been set up (user exists in DB)."""
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) AS c FROM auth_user")
    row = cur.fetchone()
    conn.close()
    return bool(row["c"])


def create_initial_user(username: str, password: str, jwt_secret: str) -> None:
    """
    Create the initial admin user in the database.
    
    Raises sqlite3.IntegrityError if username already exists.
    """
    from .hashing import hash_password
    
    salt, pwd_hash = hash_password(password)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret)
            VALUES (?, ?, ?, ?)
            """,
            (username, salt, pwd_hash, jwt_secret),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_and_secret(username: str) -> Optional[Tuple[str, str, str]]:
    """
    Get user's password salt, hash, and JWT secret.
    
    Returns (password_salt, password_hash, jwt_secret) or None if user not found.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT password_salt, password_hash, jwt_secret
            FROM auth_user
            WHERE username = ?
            """,
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["password_salt"], row["password_hash"], row["jwt_secret"]
    finally:
        conn.close()


def load_jwt_secret() -> str:
    """
    Load the JWT secret from the database.
    
    Raises RuntimeError if auth is not initialized.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT jwt_secret FROM auth_user LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Auth is not initialized. Please set up a password first.")
        return row["jwt_secret"]
    finally:
        conn.close()

