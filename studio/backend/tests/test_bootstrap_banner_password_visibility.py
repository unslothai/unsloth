# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The first-boot banner prints the generated password only when nothing else will."""

PASSWORD = "CorrectHorseBatteryStaple"
PATH = "/opt/unsloth-studio/auth/.bootstrap_password"


def _lines(**kwargs):
    from main import bootstrap_banner_lines
    return bootstrap_banner_lines("unsloth", PATH, PASSWORD, **kwargs)


def test_a_local_launch_names_the_file_and_never_the_password():
    body = "\n".join(_lines(autofill_available = True))

    assert PASSWORD not in body, "the login page fills this in; the log does not need it"
    assert f"password saved to: {PATH}" in body


def test_a_launch_without_autofill_prints_the_password():
    body = "\n".join(_lines(autofill_available = False))

    assert f"password: {PASSWORD}" in body
    # Still say where it lives: the operator may come back after the log has scrolled.
    assert f"also saved to: {PATH}" in body


def test_a_missing_password_falls_back_to_the_path():
    from main import bootstrap_banner_lines

    body = "\n".join(bootstrap_banner_lines("unsloth", PATH, None, autofill_available = False))

    assert "password: None" not in body
    assert f"password saved to: {PATH}" in body


def test_every_banner_names_the_account_and_what_to_do_next():
    for autofill in (True, False):
        body = "\n".join(_lines(autofill_available = autofill))

        assert "DEFAULT ADMIN ACCOUNT CREATED" in body
        assert "username: unsloth" in body
        assert "Open the Unsloth UI to sign in and change it." in body


def _banner_guard():
    """The `if` in the lifespan whose body prints the banner."""
    import ast
    from pathlib import Path

    import main as main_mod

    tree = ast.parse(Path(main_mod.__file__).read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "bootstrap_banner_lines" in ast.unparse(node):
            return ast.unparse(node.test)
    raise AssertionError("nothing in the lifespan prints the banner any more")


def test_the_banner_guard_does_not_key_only_on_this_call_creating_the_admin():
    """run_server's pre-bind gate calls ensure_default_admin() before the lifespan on a
    tunnel launch, so its return value is False there and a guard keyed on it alone skips
    the banner on exactly the launch that has no autofill to fall back on."""
    guard = _banner_guard()

    assert "admin_created_this_process" in guard
    assert (
        "requires_password_change" in guard
    ), "the gate can take a new password at its prompt, which retires the bootstrap one"


def test_the_creation_flag_survives_a_later_call_that_creates_nothing(monkeypatch):
    from auth import storage

    monkeypatch.setattr(storage, "_admin_created_this_process", False)
    monkeypatch.setattr(storage, "get_user_and_secret", lambda *a, **k: None)
    monkeypatch.setattr(storage, "generate_bootstrap_password", lambda: PASSWORD)
    monkeypatch.setattr(storage, "create_initial_user", lambda **kw: None)

    assert storage.ensure_default_admin() is True
    assert storage.admin_created_this_process() is True

    # The gate's call is first; the lifespan's is this one, and it creates nothing.
    monkeypatch.setattr(storage, "get_user_and_secret", lambda *a, **k: object())
    monkeypatch.setattr(storage, "_load_bootstrap_password", lambda: PASSWORD)

    assert storage.ensure_default_admin() is False
    assert storage.admin_created_this_process() is True


class _State:
    """Stands in for app.state, which is a plain namespace."""


def test_the_launch_wins_over_a_stale_api_only_environment_variable():
    """run_server sets UNSLOTH_API_ONLY and never clears it, and an embedded host may call
    run_server() again in the same process. Reading the variable would treat that second,
    normal UI launch as having no autofill and print the password into its log."""
    from main import banner_autofill_available

    state = _State()
    state.api_only = False

    assert banner_autofill_available(state, {"UNSLOTH_API_ONLY": "1"}) is True


def test_an_api_only_launch_has_no_autofill():
    from main import banner_autofill_available

    state = _State()
    state.api_only = True

    assert banner_autofill_available(state, {}) is False


def test_a_suppressed_injection_has_no_autofill():
    from main import banner_autofill_available

    state = _State()
    state.api_only = False
    state.suppress_bootstrap_injection = True

    assert banner_autofill_available(state, {}) is False


def test_a_direct_uvicorn_launch_falls_back_to_the_environment():
    """Nothing set app.state here, so the variable is all there is."""
    from main import banner_autofill_available

    assert banner_autofill_available(_State(), {"UNSLOTH_API_ONLY": "1"}) is False
    assert banner_autofill_available(_State(), {}) is True


def test_run_server_resets_the_per_launch_flags_before_the_gate():
    import ast
    from pathlib import Path

    import run as run_mod

    source = Path(run_mod.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_server"
    )
    assigns = [
        ast.unparse(node)
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign) and "app.state" in ast.unparse(node)
    ]

    assert "app.state.api_only = api_only" in assigns
    assert (
        "app.state.suppress_bootstrap_injection = False" in assigns
    ), "a sticky True from an earlier launch withholds the autofill that is available"
