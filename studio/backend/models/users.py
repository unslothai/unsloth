"""Pydantic models for user-related API endpoints.

This module defines the data models used for user authentication and management
in the FastAPI application.
"""

from pydantic import BaseModel, Field


class User(BaseModel):
    """Basic user model containing username."""

    username: str


class UserInDB(BaseModel):
    """User model with password for database storage."""

    password: str


class Token(BaseModel):
    """Authentication token model with access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str


class TokenData(BaseModel):
    """Token payload model containing username."""

    username: str | None = None

