"""
Authentication API routes
"""
from fastapi import APIRouter, HTTPException, status
import secrets

from models.auth import (
    AuthSetupRequest,
    AuthLoginRequest,
    AuthStatusResponse,
)
from models.users import Token
from auth import storage, hashing
from auth.authentication import create_access_token, reload_secret

router = APIRouter()


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status() -> AuthStatusResponse:
    """
    Check whether auth has already been initialized.

    - initialized = False -> frontend should show "Set admin password" screen.
    - initialized = True  -> frontend should show normal login.
    """
    return AuthStatusResponse(initialized=storage.is_initialized())


@router.post("/setup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def setup_auth(payload: AuthSetupRequest) -> Token:
    """
    First-time setup: create the admin user and a JWT secret.

    Can only be called once. Subsequent calls will return 400.
    """
    if storage.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth is already initialized.",
        )

    # Generate a strong random JWT secret for this installation
    jwt_secret = secrets.token_urlsafe(64)

    # Save username/password hash and secret in SQLite
    try:
        storage.create_initial_user(
            username=payload.username,
            password=payload.password,
            jwt_secret=jwt_secret,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )

    # Reload JWT secret from DB (so authentication.py picks it up)
    reload_secret()

    # Issue a token for the new user
    access_token = create_access_token(subject=payload.username)
    return Token(access_token=access_token, token_type="bearer")


@router.post("/login", response_model=Token)
async def login(payload: AuthLoginRequest) -> Token:
    """
    Login with username/password and receive a JWT.
    """
    record = storage.get_user_and_secret(payload.username)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    salt, pwd_hash, _jwt_secret = record
    if not hashing.verify_password(payload.password, salt, pwd_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = create_access_token(subject=payload.username)
    return Token(access_token=access_token, token_type="bearer")

