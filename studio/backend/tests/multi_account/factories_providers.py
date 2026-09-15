# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

PROVIDER_ID = "prov-openai-config"
MIGRATE_PROVIDER_ID = "prov-openai-migrate"
CODEX_PROVIDER_ID = "prov-codex-config"
CODEX_FLOW_ID = "prov-codex-flow"
MCP_SERVER_ID = "prov-mcp-server"
PROMPT_ENTRY_ID = "prov-prompt-entry"
PROMPT_LIST_ID = "prov-prompt-list"
RETIRABLE_USERNAME = "prov-retirable"

SENTINEL = "prov-sentinel café / 日本語"
EDITED = "prov-edited"
SAVED_API_KEY = "prov-saved-api-key"
MIGRATED_API_KEY = "prov-migrated-api-key"

# Managed accounts may only hold a public HTTP MCP address, so the probe goes to a dead public port.
MCP_URL = "http://8.8.8.8:9/mcp"

# Filled by the migrate seeder: the key must be encrypted with this process's RSA public key.
MIGRATION_BODY: dict = {"encrypted_api_key": ""}


def _encrypt_api_key(plaintext: str) -> str:
    import base64

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    from core.inference import key_exchange

    if key_exchange.get_public_key_fingerprint() is None:
        key_exchange.init_key_pair()
    public_key = serialization.load_pem_public_key(key_exchange.get_public_key_pem().encode())
    ciphertext = public_key.encrypt(
        plaintext.encode("utf-8"),
        padding.OAEP(
            mgf = padding.MGF1(algorithm = hashes.SHA256()),
            algorithm = hashes.SHA256(),
            label = None,
        ),
    )
    return base64.b64encode(ciphertext).decode("ascii")


def _create_provider(provider_id: str, provider_type: str, display_name: str) -> None:
    from core.inference.providers import get_base_url
    from storage import providers_db

    providers_db.delete_provider(provider_id)
    providers_db.create_provider(
        id = provider_id,
        provider_type = provider_type,
        display_name = display_name,
        base_url = get_base_url(provider_type),
        models = [],
        available_models = [],
    )


def _drop_in_process_flows(provider_id: str) -> None:
    from core.inference import openai_codex_auth as codex_auth
    for flow_id, flow in list(codex_auth._flows.items()):
        if flow.provider_id != provider_id:
            continue
        codex_auth._flows.pop(flow_id, None)
        for task in (flow.task, flow.cleanup_task):
            if task is not None:
                try:
                    task.cancel()
                except Exception:
                    pass
        if flow.server is not None:
            try:
                flow.server.close()
            except Exception:
                pass
            flow.server = None


@seeder("prov-provider")
def seed_provider(account) -> dict[str, str]:
    from storage import credential_secrets
    from utils.account_context import run_as

    run_as(account, _create_provider, PROVIDER_ID, "openai", SENTINEL)
    run_as(account, credential_secrets.save_provider_api_key, PROVIDER_ID, SAVED_API_KEY)
    return {"provider_id": PROVIDER_ID}


@seeder("prov-provider-without-key")
def seed_provider_without_key(account) -> dict[str, str]:
    from utils.account_context import run_as

    run_as(account, _create_provider, MIGRATE_PROVIDER_ID, "openai", SENTINEL)
    MIGRATION_BODY["encrypted_api_key"] = _encrypt_api_key(MIGRATED_API_KEY)
    return {"provider_id": MIGRATE_PROVIDER_ID}


@seeder("prov-codex-provider")
def seed_codex_provider(account) -> dict[str, str]:
    from utils.account_context import run_as

    run_as(account, _create_provider, CODEX_PROVIDER_ID, "openai_codex", SENTINEL)
    _drop_in_process_flows(CODEX_PROVIDER_ID)
    return {"provider_id": CODEX_PROVIDER_ID}


@seeder("prov-codex-connected")
def seed_codex_connected(account) -> dict[str, str]:
    from core.inference import openai_codex_auth as codex_auth
    from utils.account_context import run_as

    run_as(account, _create_provider, CODEX_PROVIDER_ID, "openai_codex", SENTINEL)
    _drop_in_process_flows(CODEX_PROVIDER_ID)
    run_as(
        account,
        codex_auth.save_oauth_bundle,
        CODEX_PROVIDER_ID,
        {
            "access_token": "prov-access-token",
            "refresh_token": "prov-refresh-token",
            "expires_at": 4102444800,
            "account_id": "prov-chatgpt-account",
        },
    )
    return {"provider_id": CODEX_PROVIDER_ID}


@seeder("prov-codex-flow")
def seed_codex_flow(account) -> dict[str, str]:
    import time

    from core.inference import openai_codex_auth as codex_auth
    from utils.account_context import run_as

    run_as(account, _create_provider, CODEX_PROVIDER_ID, "openai_codex", SENTINEL)
    _drop_in_process_flows(CODEX_PROVIDER_ID)
    now = time.time()
    flow = codex_auth.OAuthFlow(
        id = CODEX_FLOW_ID,
        provider_id = CODEX_PROVIDER_ID,
        method = "browser",
        created_at = now,
        expires_at = now + 900,
        state = "prov-oauth-state",
        verifier = "prov-oauth-verifier",
        redirect_uri = "http://localhost:1455/auth/callback",
        authorization_url = "https://auth.openai.com/oauth/authorize?state=prov-oauth-state",
        status = "pending",
        marker = "prov-oauth-marker",
    )
    run_as(
        account,
        codex_auth.save_oauth_flow_marker,
        CODEX_PROVIDER_ID,
        "prov-oauth-marker",
        flow,
    )
    return {"provider_id": CODEX_PROVIDER_ID, "flow_id": CODEX_FLOW_ID}


@seeder("prov-mcp-refresh")
def seed_mcp_refresh(account) -> dict[str, str]:
    from storage import mcp_servers_db
    from utils.account_context import run_as

    run_as(
        account,
        mcp_servers_db.create_server,
        MCP_SERVER_ID,
        SENTINEL,
        MCP_URL,
        headers_json = None,
        is_enabled = False,
        use_oauth = False,
    )
    return {"server_id": MCP_SERVER_ID}


@seeder("prov-prompt-entry")
def seed_prompt_entry(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    run_as(
        account,
        studio_db.upsert_prompt_entry,
        {
            "id": PROMPT_ENTRY_ID,
            "name": SENTINEL,
            "text": SENTINEL,
            "createdAt": 1000,
            "updatedAt": 1000,
        },
    )
    return {"entry_id": PROMPT_ENTRY_ID}


@seeder("prov-prompt-list")
def seed_prompt_list(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    run_as(
        account,
        studio_db.upsert_prompt_list,
        {
            "id": PROMPT_LIST_ID,
            "name": SENTINEL,
            "items": [PROMPT_ENTRY_ID],
            "createdAt": 1000,
            "updatedAt": 1000,
        },
    )
    return {"list_id": PROMPT_LIST_ID}


@seeder("prov-account")
def seed_account(account) -> dict[str, str]:
    return {"account_id": account.account_id}


@seeder("prov-retirable-account")
def seed_retirable_account(account) -> dict[str, str]:
    """A managed account of its own, whose deletion retires a private root the matrix's unchanged-workspace assertion would misread as erased."""
    from auth import storage
    from storage import studio_db
    from utils.account_context import AccountContext, run_as

    for row in storage.list_accounts():
        if row["username"] == RETIRABLE_USERNAME:
            storage.delete_account(row["account_id"], lambda _context: None)
    issued = storage.issue_account_setup_code(username = RETIRABLE_USERNAME)
    account_id = issued["account"]["account_id"]
    # First database use creates the private root, so the retire has a directory to move.
    run_as(AccountContext(account_id, RETIRABLE_USERNAME, "user"), studio_db.list_prompt_entries)
    return {"account_id": account_id}


PROMPT_ENTRY_BODY = {
    "id": PROMPT_ENTRY_ID,
    "name": EDITED,
    "text": EDITED,
    "createdAt": 1000,
    "updatedAt": 2000,
}
PROMPT_LIST_BODY = {
    "id": PROMPT_LIST_ID,
    "name": EDITED,
    "items": [PROMPT_ENTRY_ID],
    "createdAt": 1000,
    "updatedAt": 2000,
}

_PROMPT_UPSERT_REASON = (
    "PUT is an upsert keyed only by id inside the caller's own database, so a foreign caller "
    "writes its own row and never sees alice's; the snapshot proves hers is untouched"
)
_PROMPT_DELETE_REASON = (
    "DELETE is an idempotent no-op scoped to the caller's own database, so a foreign caller "
    "gets 204 after deleting nothing; the snapshot proves alice's row survived"
)
_PROVIDER_DELETE_OWNER_REASON = (
    "the owner is not a managed account, so its delete falls through to the idempotent path "
    "and removes nothing from its own database"
)
_ACCOUNTS_DELETE_REASON = (
    "owner administration route behind require_owner, inverted like its siblings; the target is "
    "a disposable account because a real owner delete renames the target's whole private root"
)
_ACCOUNTS_SELF_REASON = (
    "owner administration route behind require_owner, so the contract inverts: the owner "
    "succeeds against alice, both managed accounts are refused, and it cannot administer itself"
)

FACTORIES = {
    "routes.providers:PUT:/{provider_id}": Factory(
        "prov-provider", {"display_name": EDITED}, fragment = EDITED
    ),
    "routes.providers:DELETE:/{provider_id}": Factory(
        "prov-provider",
        success = 204,
        owner = (204,),
        reason = _PROVIDER_DELETE_OWNER_REASON,
    ),
    "routes.providers:PUT:/{provider_id}/api-key/migrate": Factory(
        "prov-provider-without-key",
        MIGRATION_BODY,
        fragment = '"has_api_key":true',
    ),
    "routes.openai_codex_auth:GET:/{provider_id}/codex/models": Factory(
        "prov-codex-provider", fragment = '"source":"curated"'
    ),
    "routes.openai_codex_auth:POST:/{provider_id}/oauth/start": Factory(
        "prov-codex-provider",
        {"method": "browser"},
        fragment = '"status":"pending"',
    ),
    "routes.openai_codex_auth:DELETE:/{provider_id}/oauth": Factory(
        "prov-codex-connected", success = 204
    ),
    "routes.openai_codex_auth:GET:/{provider_id}/oauth/flows/{flow_id}": Factory(
        "prov-codex-flow", fragment = CODEX_FLOW_ID
    ),
    "routes.openai_codex_auth:DELETE:/{provider_id}/oauth/flows/{flow_id}": Factory(
        "prov-codex-flow", success = 204
    ),
    "routes.mcp_servers:POST:/{server_id}/refresh": Factory(
        "prov-mcp-refresh", fragment = '"ok":false'
    ),
    "routes.prompts:PUT:/entries/{entry_id}": Factory(
        "prov-prompt-entry",
        PROMPT_ENTRY_BODY,
        fragment = EDITED,
        owner = (200,),
        wrong = (200,),
        reason = _PROMPT_UPSERT_REASON,
    ),
    "routes.prompts:DELETE:/entries/{entry_id}": Factory(
        "prov-prompt-entry",
        success = 204,
        owner = (204,),
        wrong = (204,),
        reason = _PROMPT_DELETE_REASON,
    ),
    "routes.prompts:PUT:/lists/{list_id}": Factory(
        "prov-prompt-list",
        PROMPT_LIST_BODY,
        fragment = EDITED,
        owner = (200,),
        wrong = (200,),
        reason = _PROMPT_UPSERT_REASON,
    ),
    "routes.prompts:DELETE:/lists/{list_id}": Factory(
        "prov-prompt-list",
        success = 204,
        owner = (204,),
        wrong = (204,),
        reason = _PROMPT_DELETE_REASON,
    ),
    "routes.accounts:PATCH:/{account_id}": Factory(
        "prov-account",
        {"is_active": False},
        owner = (200,),
        right = (403,),
        wrong = (403,),
        self_expected = (400,),
        reason = _ACCOUNTS_SELF_REASON,
    ),
    "routes.accounts:DELETE:/{account_id}": Factory(
        "prov-retirable-account",
        success = 204,
        owner = (204,),
        right = (403,),
        wrong = (403,),
        reason = _ACCOUNTS_DELETE_REASON,
    ),
    "routes.accounts:POST:/{account_id}/setup-code": Factory(
        "prov-account",
        owner = (200,),
        right = (403,),
        wrong = (403,),
        self_expected = (400,),
        reason = _ACCOUNTS_SELF_REASON,
    ),
}

SKIPPED = {
    "routes.openai_codex_auth:POST:/{provider_id}/oauth/flows/{flow_id}/complete": (
        "completing a flow exchanges the pasted authorization code with auth.openai.com"
    ),
}
