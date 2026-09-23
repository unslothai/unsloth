# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Credential-safe urllib redirects. Stdlib-only: the standalone installers import it
without reaching the rest of the backend."""

import urllib.parse
import urllib.request


class AuthSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep credentials on their original origin and refuse TLS downgrades."""

    @staticmethod
    def _origin(url):
        parts = urllib.parse.urlsplit(url)
        scheme = parts.scheme.lower()
        try:
            port = parts.port
        except ValueError:
            # A fresh object(), never a constant: an unreadable port must compare unequal
            # to every origin including itself, and no caller here catches ValueError.
            return scheme, (parts.hostname or "").lower(), object()
        if port is None:
            port = 443 if scheme == "https" else 80
        return scheme, (parts.hostname or "").lower(), port

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = self._origin(req.full_url)
        new = self._origin(newurl)
        if old[0] == "https" and new[0] == "http":
            return None
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and new != old:
            result.headers.pop("Authorization", None)
        return result


def auth_safe_open(req, timeout):
    """Open a request without forwarding credentials to another origin."""
    return urllib.request.build_opener(AuthSafeRedirectHandler()).open(req, timeout=timeout)
