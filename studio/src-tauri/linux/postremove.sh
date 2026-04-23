#!/bin/sh
# Post-removal script for Unsloth Studio (deb/rpm)
# Runs non-interactively; never deletes user data or touches other users' homes.

case "${1:-}" in
    upgrade|1|2) exit 0 ;;
esac

exit 0
