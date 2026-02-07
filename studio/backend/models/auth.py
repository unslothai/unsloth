"""
Pydantic schemas for Authentication API
"""
from pydantic import BaseModel, Field


class AuthSetupRequest(BaseModel):
    """First-time setup: create the initial admin user + password."""
    username: str = Field(..., description="Admin username")
    password: str = Field(..., min_length=8, description="Admin password (minimum 8 characters)")


class AuthLoginRequest(BaseModel):
    """Login payload: username/password to obtain a JWT."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class AuthStatusResponse(BaseModel):
    """Indicate whether auth has been initialized."""
    initialized: bool = Field(..., description="True if auth setup has been completed")

