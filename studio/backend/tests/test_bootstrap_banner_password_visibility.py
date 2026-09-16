# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The first-boot banner prints the generated password only when nothing else will.

``_inject_bootstrap`` normally fills the login form in for the operator, so the banner
names the file and keeps the credential out of the log. When the injection is
suppressed (a public URL) or no page is served at all (``--api-only``), the file is the
only copy and the banner has to say it.
"""

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
