# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hub + Download Manager feature module.

Self-contained routes, schemas, utilities, workers, and storage for the model
inventory layer and the HuggingFace download manager. Wired into the FastAPI
app via two routers plus startup/shutdown hooks in main.py."""
