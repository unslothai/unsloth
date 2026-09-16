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

    body = "\n".join(
        bootstrap_banner_lines("unsloth", PATH, None, autofill_available = False)
    )

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
    assert "requires_password_change" in guard, (
        "the gate can take a new password at its prompt, which retires the bootstrap one"
    )


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
