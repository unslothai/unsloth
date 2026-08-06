# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio errors that are safe to show a client verbatim.

``safe_error_detail`` flattens every exception to "An internal error occurred" so a
raw ``str(e)`` cannot leak a path. Right for a failure, wrong for a capability
answer, where the reason is the whole reply: the Audio page reported an internal
error for a model that had loaded fine and simply cannot generate on this host.
These carry no path or user input, so the route passes them straight through.
"""

from __future__ import annotations


# Tagged on the worker's audio_error payload so the parent recognises the case
# without matching on prose.
AUDIO_UNSUPPORTED_CODE = "audio_unsupported_backend"


class AudioBackendUnsupportedError(RuntimeError):
    """The model loaded fine; this backend cannot do this audio task.

    Unlike a generation failure, no retry, shorter input or freed memory helps.
    """

    def __init__(
        self,
        detail: str,
        *,
        hint: str | None = None,
    ):
        self.detail = detail
        self.hint = hint
        super().__init__(detail if not hint else f"{detail} {hint}")

    @property
    def message(self) -> str:
        return self.args[0] if self.args else self.detail
