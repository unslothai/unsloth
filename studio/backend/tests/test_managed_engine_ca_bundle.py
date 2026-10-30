# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""A launched engine verifies TLS against Studio's certifi bundle, not the one its lock pinned."""

from __future__ import annotations

import inspect

import certifi
import pytest

from core.inference import managed_engine


def test_an_engine_without_a_chosen_bundle_gets_studios_certifi():
    env = {"PATH": "/usr/bin"}
    managed_engine._point_tls_at_studio_ca_bundle(env)
    assert env["SSL_CERT_FILE"] == certifi.where()
    assert env["REQUESTS_CA_BUNDLE"] == certifi.where()
    before = dict(env)
    managed_engine._point_tls_at_studio_ca_bundle(env)
    assert env == before


@pytest.mark.parametrize(
    "name", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
)
def test_a_bundle_the_user_chose_is_left_alone(name):
    env = {name: "/etc/corp/ca.pem"}
    managed_engine._point_tls_at_studio_ca_bundle(env)
    assert env == {name: "/etc/corp/ca.pem"}


def test_a_missing_bundle_file_sets_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(certifi, "where", lambda: str(tmp_path / "gone.pem"))
    env = {}
    managed_engine._point_tls_at_studio_ca_bundle(env)
    assert env == {}


def test_the_local_launch_applies_it():
    source = inspect.getsource(managed_engine)
    assert source.count("_point_tls_at_studio_ca_bundle(child_env)") == 1
