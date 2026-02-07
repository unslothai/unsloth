"""
Authentication module for JWT-based auth with SQLite storage.
"""
from .authentication import create_access_token, get_current_subject, reload_secret
from .storage import (
    is_initialized,
    create_initial_user,
    get_user_and_secret,
    load_jwt_secret,
)
from .hashing import hash_password, verify_password

__all__ = [
    "create_access_token",
    "get_current_subject",
    "reload_secret",
    "is_initialized",
    "create_initial_user",
    "get_user_and_secret",
    "load_jwt_secret",
    "hash_password",
    "verify_password",
]

