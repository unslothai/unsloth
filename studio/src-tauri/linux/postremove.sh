#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Post-removal script for the Unsloth Studio Debian package
# Runs non-interactively; never deletes user data or touches other users' homes.

case "${1:-}" in
    upgrade|1|2) exit 0 ;;
esac

exit 0
