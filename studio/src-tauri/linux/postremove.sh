#!/bin/sh
# Post-removal script for Unsloth Studio (deb/rpm)
# Offers to clean up ~/.unsloth data directories.

# Only prompt on full removal (not upgrade). Debian passes "remove" or "purge";
# RPM runs this on erase (upgrade passes the install count as $1 > 0).
case "${1:-}" in
    upgrade|1|2) exit 0 ;;
esac

# When run as root during package removal, $HOME is /root.
# Check all real user home directories instead.
cleaned=false
for user_home in /home/*; do
    [ -d "$user_home/.unsloth" ] || continue
    user_name=$(basename "$user_home")
    printf "\nFound Unsloth data at %s/.unsloth\n" "$user_home"
    printf "Remove all data for user '%s'? (models, training outputs, configuration)\n" "$user_name"
    printf "[y/N] "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS])
            rm -rf "$user_home/.unsloth"
            printf "Removed %s/.unsloth\n" "$user_home"
            cleaned=true
            ;;
        *)
            printf "Kept %s/.unsloth\n" "$user_home"
            ;;
    esac
done

# Also check /root in case someone ran it as root
if [ -d "/root/.unsloth" ]; then
    printf "\nFound Unsloth data at /root/.unsloth\n"
    printf "Remove? [y/N] "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS])
            rm -rf "/root/.unsloth"
            printf "Removed /root/.unsloth\n"
            ;;
    esac
fi
