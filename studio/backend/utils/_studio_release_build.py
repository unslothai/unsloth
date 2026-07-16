# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build-stamped Studio release metadata.

Release builds may rewrite this module in the build workspace before creating
Python artifacts. Keep the committed value neutral so source checkouts do not
accidentally report a stale release tag.
"""

STUDIO_RELEASE_VERSION = None
