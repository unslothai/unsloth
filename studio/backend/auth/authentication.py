import secrets
from datetime import UTC, datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt

from .storage import load_jwt_secret

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Load stable secret from SQLite (set during first-time setup)
# This will raise RuntimeError if auth hasn't been initialized yet
try:
    SECRET_KEY = load_jwt_secret()
except RuntimeError:
    # Fallback: use a temporary secret until setup is complete
    # This allows the app to start, but protected routes will fail until setup
    SECRET_KEY = secrets.token_urlsafe(64)

security = HTTPBearer()  # Reads Authorization: Bearer <token>


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT for the given subject (e.g. username).

    Tokens are valid across restarts because SECRET_KEY is stored in SQLite.
    """
    to_encode = {"sub": subject}
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def reload_secret() -> None:
    """
    Reload the JWT secret from SQLite.
    
    Call this after setup to ensure new tokens use the persistent secret.
    """
    global SECRET_KEY
    SECRET_KEY = load_jwt_secret()


async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.

    Use this as a dependency on routes that should be protected, e.g.:

        @router.get("/secure")
        async def secure_endpoint(current_subject: str = Depends(get_current_subject)):
            ...
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return subject
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
# token = create_access_token("local-user")
# print(token)


