# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Data Recipe core (DataDesigner wrapper + job runner).
"""

from .jobs import JobManager, get_job_manager

__all__ = ["JobManager", "get_job_manager"]
