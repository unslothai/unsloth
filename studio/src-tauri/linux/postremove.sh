#!/bin/sh
# Post-removal script for Unsloth Studio (deb/rpm)
# Offers to clean up the ~/.unsloth data directory.

# Only prompt on full removal (not upgrade). Debian passes "remove" or "purge";
# RPM runs this on erase (upgrade passes the install count as $1 > 0).
case "${1:-}" in
    upgrade|1|2) exit 0 ;;
esac

UNSLOTH_DIR="$HOME/.unsloth"
if [ -d "$UNSLOTH_DIR" ]; then
    printf "\nUnsloth Studio has been removed.\n"
    printf "Remove all Unsloth data (%s)?\n" "$UNSLOTH_DIR"
    printf "This deletes installed models, training outputs, and configuration.\n"
    printf "[y/N] "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS])
            rm -rf "$UNSLOTH_DIR"
            printf "Removed %s\n" "$UNSLOTH_DIR"
            ;;
        *)
            printf "Kept %s\n" "$UNSLOTH_DIR"
            ;;
    esac
fi
