#!/bin/sh
# Post-removal script for the Unsloth Studio Debian package
# Runs non-interactively; never deletes user data or touches other users' homes.

case "${1:-}" in
    upgrade|1|2) exit 0 ;;
esac

exit 0
