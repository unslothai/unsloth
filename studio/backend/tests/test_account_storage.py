# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise owner and managed storage in the same interpreter."""

import importlib
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from storage import api_usage_db, profile_stats_db, studio_db
from utils.account_context import OWNER, AccountContext, account_thread, current_account_id, run_as
from utils.paths import storage_roots as roots
from utils.paths.lazy import LazyPath

ALICE = AccountContext("11111111111111111111111111111111", "alice")
BOB = AccountContext("22222222222222222222222222222222", "bob")
ACCOUNTS = (OWNER, ALICE, BOB)


@pytest.fixture
def account_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    studio_db.close_wal_keeper()
    profile_stats_db._cache.clear()
    memos = []
    for name in (
        "model_memory_settings",
        "vram_budget_settings",
        "openai_auto_switch_settings",
        "embedding_model_settings",
    ):
        module = importlib.import_module(f"utils.{name}")
        for attribute in ("_cache", "_cached", "_generation", "_resolved_gguf_memo"):
            memo = getattr(module, attribute, None)
            if isinstance(memo, dict):
                memo.clear()
                memos.append(memo)
    yield tmp_path
    for memo in memos:
        memo.clear()
    studio_db.close_wal_keeper()
    profile_stats_db._cache.clear()


@pytest.mark.parametrize(
    "module_name,table",
    [
        ("studio_db", "app_settings"),
        ("rag_db", "documents"),
        ("providers_db", "llm_providers"),
        ("mcp_servers_db", "mcp_servers"),
        ("credential_secrets", "credential_secrets"),
        ("chat_generation_runs_db", "chat_generation_runs"),
    ],
)
def test_each_database_gets_its_schema(account_home, module_name, table):
    module = importlib.import_module(f"storage.{module_name}")
    module.reset_schema_state_for_tests()
    connect = module._connect if module_name == "chat_generation_runs_db" else module.get_connection
    for account in ACCOUNTS:
        conn = run_as(account, connect)
        try:
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
            if module_name == "chat_generation_runs_db":
                conn.execute("SELECT progress_at, progress_tokens FROM chat_generation_runs")
            # A real persistent row proves each connection is a separate file.
            conn.execute("CREATE TABLE account_probe (account_id TEXT)")
            conn.execute("INSERT INTO account_probe VALUES (?)", (account.account_id,))
            conn.commit()
        finally:
            conn.close()
    for account in ACCOUNTS:
        conn = run_as(account, connect)
        try:
            assert (
                conn.execute("SELECT account_id FROM account_probe").fetchall()[0][0]
                == account.account_id
            )
        finally:
            conn.close()
    assert len(module._schema_ready) == 3
    assert all(path.is_absolute() for path in module._schema_ready)
    module.reset_schema_state_for_tests()
    assert module._schema_ready == set()


def test_wal_keepers_can_close_one_account(account_home):
    for account in ACCOUNTS:
        assert run_as(account, studio_db.open_wal_keeper)
    paths = {account: run_as(account, roots.studio_db_path) for account in ACCOUNTS}
    alice = studio_db._wal_keepers[paths[ALICE]]
    studio_db.close_wal_keeper_for(paths[ALICE])
    with pytest.raises(sqlite3.ProgrammingError, match = "closed"):
        alice.execute("SELECT 1")
    for account in (OWNER, BOB):
        assert studio_db._wal_keepers[paths[account]].execute("SELECT 1").fetchone()[0] == 1
    studio_db.close_wal_keeper()
    assert studio_db._wal_keepers == {}


def _receipt() -> api_usage_db.ApiUsageReceipt:
    return api_usage_db.ApiUsageReceipt(
        id = "same-receipt",
        subject = "same-subject",
        endpoint = "/v1/chat/completions",
        model = "model",
        status = "completed",
        prompt_tokens = 2,
        completion_tokens = 3,
        total_tokens = 5,
        created_at = 1,
    )


def test_usage_writer_retains_account_through_retries(account_home, monkeypatch):
    seen = []
    retried = False

    def sink(receipt):
        nonlocal retried
        seen.append(current_account_id())
        if current_account_id() == ALICE.account_id and not retried:
            retried = True
            raise sqlite3.OperationalError("database is locked")
        return api_usage_db.record_api_usage(receipt)

    monkeypatch.setattr(api_usage_db, "_sleep_after_busy", lambda delay: None)
    writer = api_usage_db.ApiUsageWriter(sink = sink)
    try:
        for account in ACCOUNTS:
            assert run_as(account, writer.submit, _receipt())
    finally:
        assert writer.stop(timeout = 10)
    assert seen == [OWNER.account_id, ALICE.account_id, ALICE.account_id, BOB.account_id]
    for account in ACCOUNTS:
        conn = run_as(account, studio_db.get_connection)
        try:
            assert conn.execute("SELECT sum(total_tokens) FROM api_usage_events").fetchone()[0] == 5
        finally:
            conn.close()


def test_profile_cache_and_invalidation_are_private(account_home):
    # Empty databases have identical fingerprints; one account's cached object
    # must not be served to another even in that case.
    values = {
        account: run_as(account, profile_stats_db.compute_profile_stats) for account in ACCOUNTS
    }
    assert values[OWNER] is not values[ALICE]
    assert values[ALICE] is not values[BOB]
    for account in ACCOUNTS:
        assert run_as(account, profile_stats_db.compute_profile_stats) is values[account]
    run_as(ALICE, profile_stats_db.invalidate_profile_stats_cache)
    assert run_as(OWNER, profile_stats_db.compute_profile_stats) is values[OWNER]
    assert run_as(BOB, profile_stats_db.compute_profile_stats) is values[BOB]
    assert run_as(ALICE, profile_stats_db.compute_profile_stats) is not values[ALICE]


@pytest.mark.parametrize(
    "module_name,setter,getter,first,second",
    [
        (
            "model_memory_settings",
            "set_model_memory_settings",
            "get_keep_resident",
            (True, False),
            (False, False),
        ),
        (
            "vram_budget_settings",
            "set_vram_budget_fraction",
            "get_vram_budget_fraction",
            (0.8,),
            (0.9,),
        ),
        (
            "openai_auto_switch_settings",
            "set_openai_auto_switch",
            "get_openai_auto_switch_enabled",
            (True, 0),
            (False, 0),
        ),
        (
            "embedding_model_settings",
            "set_rag_embedding_model",
            "get_stored_embedding_model",
            ("org/one",),
            ("org/two",),
        ),
    ],
)
def test_settings_memos_follow_accounts(
    account_home, monkeypatch, module_name, setter, getter, first, second
):
    module = importlib.import_module(f"utils.{module_name}")
    monkeypatch.setattr(module, "_CACHE_TTL_S", 3600)
    for account, args in ((OWNER, first), (ALICE, second), (BOB, first)):
        run_as(account, getattr(module, setter), *args)
        assert run_as(account, getattr(module, getter)) == args[0]
    # Reads hit the memos while a different account's last value is still hot.
    for account, expected in ((OWNER, first[0]), (ALICE, second[0]), (BOB, first[0])):
        assert run_as(account, getattr(module, getter)) == expected
    run_as(ALICE, getattr(module, setter), *first)
    assert run_as(OWNER, getattr(module, getter)) == first[0]


def test_embedding_resolution_memo_is_private(account_home):
    from utils import embedding_model_settings as settings

    settings._resolved_gguf_memo.clear()
    for account, repo in ((OWNER, "org/owner-GGUF"), (ALICE, "org/alice-GGUF")):
        run_as(account, settings.set_rag_embedding_model, "org/embed", gguf_repo = repo)
        assert run_as(account, settings.get_stored_gguf_repo, "org/embed") == repo
        run_as(account, settings.set_rag_embedding_model, "org/other")
    assert run_as(OWNER, settings.remembered_gguf_repo, "org/embed") == "org/owner-GGUF"
    assert run_as(ALICE, settings.remembered_gguf_repo, "org/embed") == "org/alice-GGUF"
    assert run_as(BOB, settings.remembered_gguf_repo, "org/embed") is None
    settings._resolved_gguf_memo.clear()


def test_hf_validation_cache_and_budget_are_private(monkeypatch):
    from utils import hf_token_validation as validation

    validation.reset_hf_token_validation_state()
    calls = []
    monkeypatch.setattr(
        validation,
        "_check_remote",
        lambda token: calls.append(current_account_id())
        or validation.TokenValidationResult(status = "valid"),
    )
    monkeypatch.setattr(validation, "_MAX_ATTEMPTS", 1)
    try:
        for account in ACCOUNTS:
            assert (
                run_as(
                    account, validation.validate_hf_token, "same-token", rate_key = "same-client"
                ).status
                == "valid"
            )
            assert (
                run_as(
                    account, validation.validate_hf_token, "same-token", rate_key = "same-client"
                ).status
                == "valid"
            )
            assert (
                run_as(
                    account, validation.validate_hf_token, "another-token", rate_key = "same-client"
                ).status
                == "rate_limited"
            )
        assert calls == [account.account_id for account in ACCOUNTS]
    finally:
        validation.reset_hf_token_validation_state()


def test_hub_reexports_the_canonical_owner_and_account_roots(account_home):
    from hub.utils import paths as hub
    expected = {
        "studio_root": account_home,
        "cache_root": account_home / "cache",
        "assets_root": account_home / "assets",
        "datasets_root": account_home / "assets/datasets",
        "dataset_uploads_root": account_home / "assets/datasets/uploads",
        "recipe_datasets_root": account_home / "assets/datasets/recipes",
        "outputs_root": account_home / "outputs",
        "exports_root": account_home / "exports",
        "tmp_root": roots.tmp_root(),
    }
    for name, owner_path in expected.items():
        assert getattr(hub, name) is getattr(roots, name)
        assert getattr(hub, name)() == owner_path
        assert run_as(ALICE, getattr(hub, name)) == run_as(ALICE, getattr(roots, name))


def test_lazy_paths_keep_imported_references_live(account_home):
    path = LazyPath(roots.outputs_root)
    for account in ACCOUNTS:
        expected = run_as(account, roots.outputs_root)
        assert run_as(account, path) == expected
        assert run_as(account, os.fspath, path) == str(expected)
        assert run_as(account, str, path) == str(expected)
        assert run_as(account, lambda: path / "child") == expected / "child"
        assert run_as(account, lambda: path == expected)
        assert run_as(account, lambda: path.parent) == expected.parent
    assert not (account_home / "outputs").exists()


@pytest.mark.parametrize(
    "filename,names",
    [
        (
            "hub/services/datasets/local.py",
            {
                "LOCAL_DATASETS_ROOT": roots.recipe_datasets_root,
                "DATASET_UPLOAD_DIR": roots.dataset_uploads_root,
            },
        ),
        ("core/data_recipe/jobs/worker.py", {"_ARTIFACT_ROOT": roots.recipe_datasets_root}),
        (
            "routes/data_recipe/seed.py",
            {
                "SEED_UPLOAD_DIR": roots.seed_uploads_root,
                "UNSTRUCTURED_UPLOAD_ROOT": roots.unstructured_uploads_root,
            },
        ),
        (
            "plugins/data-designer-unstructured-seed/src/data_designer_unstructured_seed/chunking.py",
            {"_CACHE_DIR": roots.unstructured_seed_cache_root},
        ),
    ],
)
def test_legacy_root_names_resolve_at_use(account_home, filename, names):
    # Load only the root declarations, avoiding optional plugin engine imports.
    import ast

    source = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(source.read_text())
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    namespace = {"LazyPath": LazyPath, **{fn.__name__: fn for fn in names.values()}}
    exec(compile(ast.Module(body = assignments, type_ignores = []), str(source), "exec"), namespace)
    for name, root in names.items():
        accessor = namespace[name]
        for account in ACCOUNTS:
            assert run_as(account, Path, accessor) == run_as(account, root)


def test_checkpoint_defaults_resolve_in_the_calling_account(account_home, monkeypatch):
    from utils.models import checkpoints

    seen = []
    monkeypatch.setattr(checkpoints, "scan_checkpoints", lambda path: seen.append(path) or [])
    for account in ACCOUNTS:
        assert run_as(account, checkpoints.list_preview_targets) == []
    assert seen == [str(run_as(account, roots.outputs_root)) for account in ACCOUNTS]


def test_owner_setting_cache_hits_do_not_query_storage(account_home, monkeypatch):
    from utils import (
        embedding_model_settings,
        model_memory_settings,
        openai_auto_switch_settings,
        vram_budget_settings,
    )

    getters = (
        model_memory_settings.get_keep_resident,
        vram_budget_settings.get_vram_budget_fraction,
        openai_auto_switch_settings.get_openai_auto_switch_enabled,
        embedding_model_settings.get_stored_embedding_model,
    )
    expected = [get() for get in getters]

    def unexpected(*args, **kwargs):
        pytest.fail("A hot owner setting lookup queried storage")

    monkeypatch.setattr(studio_db, "get_app_setting", unexpected)
    monkeypatch.setattr(studio_db, "get_app_settings", unexpected)
    for _ in range(10):
        assert [get() for get in getters] == expected


def test_hf_inflight_checks_do_not_join_another_account(monkeypatch):
    from utils import hf_token_validation as validation

    validation.reset_hf_token_validation_state()
    entered = threading.Barrier(2)
    results = []

    def remote(token):
        entered.wait(timeout = 5)
        return validation.TokenValidationResult(status = "valid")

    monkeypatch.setattr(validation, "_check_remote", remote)
    monkeypatch.setattr(validation, "_INFLIGHT_WAIT_SECONDS", 0.1)

    def check():
        results.append(validation.validate_hf_token("same-token", rate_key = "same-client").status)

    threads = [account_thread(account = account, target = check) for account in (ALICE, BOB)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout = 6)
        assert results == ["valid", "valid"]
    finally:
        validation.reset_hf_token_validation_state()


def test_local_dataset_list_uses_the_acting_accounts_uploads(account_home):
    from hub.services.datasets import local
    for account in ACCOUNTS:
        directory = run_as(account, roots.dataset_uploads_root)
        directory.mkdir(parents = True)
        (directory / f"{account.username}.csv").write_text("value\n1\n")
    for account in ACCOUNTS:
        items = run_as(account, local._build_uploaded_dataset_items)
        assert [item.label for item in items] == [f"{account.username}.csv"]


def test_scan_checkpoints_default_uses_the_acting_account(account_home, monkeypatch):
    from utils.models import checkpoints

    seen = []

    def resolve(path):
        seen.append(path)
        return Path(path)

    monkeypatch.setattr(checkpoints, "resolve_output_dir", resolve)
    for account in ACCOUNTS:
        assert run_as(account, checkpoints.scan_checkpoints) == []
    assert seen == [str(run_as(account, roots.outputs_root)) for account in ACCOUNTS]


@pytest.mark.parametrize(
    "module_name",
    [
        "studio_db",
        "rag_db",
        "providers_db",
        "mcp_servers_db",
        "credential_secrets",
        "chat_generation_runs_db",
    ],
)
def test_warm_owner_connections_do_not_resolve_database_again(
    account_home, monkeypatch, module_name
):
    module = importlib.import_module(f"storage.{module_name}")
    connect = module._connect if module_name == "chat_generation_runs_db" else module.get_connection
    path_name = "rag_db_path" if module_name == "rag_db" else "studio_db_path"
    path = getattr(module, path_name)()
    monkeypatch.setattr(module, path_name, lambda: path)
    connect().close()
    original = Path.resolve

    def resolve(candidate, *args, **kwargs):
        assert (
            candidate != path
        ), "Warm owner connections must not repeat database realpath resolution"
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    connect().close()
