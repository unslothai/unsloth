"""CLI desktop-secret provisioning must delegate to backend storage, not reimplement it."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_STUDIO_PY = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"


def _function_body(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.index("\n\n\n", start)
    return source[start:end]


def test_create_desktop_secret_delegates_to_storage():
    src = CLI_STUDIO_PY.read_text()
    body = _function_body(src, "_create_desktop_secret_in_cli")
    # Must call backend storage, not re-implement PBKDF2.
    assert "storage.create_desktop_secret" in body
    # Drift-risk markers that previously lived in this function must be gone.
    for forbidden in ("pbkdf2_hmac", "token_urlsafe", "INSERT OR REPLACE"):
        assert forbidden not in body, f"leftover duplicated logic: {forbidden}"


def test_connect_auth_db_uses_storage_db_path():
    src = CLI_STUDIO_PY.read_text()
    body = _function_body(src, "_connect_auth_db")
    # Must read the authoritative path from storage, not compute locally.
    assert "storage.DB_PATH" in body
    assert "storage.ensure_schema" in body
    # Hardcoded local path must be gone.
    assert '"auth.db"' not in body
    assert "auth_dir / \"auth.db\"" not in body


def test_cli_no_longer_has_duplicated_hashing_helpers():
    src = CLI_STUDIO_PY.read_text()
    # The reimplemented helpers that drifted from backend storage.
    for helper in (
        "def _pbkdf2_hex",
        "def _hash_password",
        "def _get_or_create_api_key_pbkdf2_salt",
        "def _ensure_cli_default_admin",
    ):
        assert helper not in src, f"{helper} should be removed in favor of backend storage"


def test_cli_no_longer_declares_pbkdf2_constants():
    src = CLI_STUDIO_PY.read_text()
    # Module-level constants that mirrored storage.py -- they must live in one place only.
    for constant in (
        "PBKDF2_ITERATIONS",
        "API_KEY_PBKDF2_SALT_KEY",
        "DESKTOP_SECRET_HASH_KEY",
        "DESKTOP_SECRET_CREATED_AT_KEY",
    ):
        assert constant not in src, f"{constant} should not be redeclared in the CLI"
