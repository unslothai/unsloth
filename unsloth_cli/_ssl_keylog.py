# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys


def drop_unwritable_ssl_keylog_file():
    """Unset SSLKEYLOGFILE when this process cannot append to it.

    ssl.create_default_context opens that path for every context it builds, so an
    unwritable one makes every HTTPS client raise: httpx at construction, which
    kills the Studio backend on import, and requests on every call. Some security
    software sets it machine-wide to a path no user process can open.
    """
    path = os.environ.get("SSLKEYLOGFILE")
    if not path:
        return False
    try:
        with open(path, "a", encoding = "utf-8"):
            pass
        return False
    except OSError as exc:
        os.environ.pop("SSLKEYLOGFILE", None)
        try:
            print(
                f"Unsloth: ignoring SSLKEYLOGFILE={path!r}, which this process cannot "
                f"write ({type(exc).__name__}); TLS key logging is off.",
                file = sys.stderr,
            )
        except Exception:
            pass
        return True
