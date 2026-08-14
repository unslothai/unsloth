#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 TOOL_CACHE_DIR" >&2
  exit 2
fi

tools_dir="$1"
mkdir -p "$tools_dir"

script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
LINUXDEPLOY_URL="https://github.com/tauri-apps/binary-releases/releases/download/linuxdeploy/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_SHA256="e762bea85c8eb0d4b3508d46e5c1f037f717d0f9303ae3b4aafc8b04991fa1ef"
GTK_PLUGIN_URL="https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gtk/b5eb8d05b4c0ed40107fe2158c5d8527f94568ef/linuxdeploy-plugin-gtk.sh"
GTK_PLUGIN_SHA256="cb379f9b0733e9ad9f8bd78f8c2fa038aef2478523bb7d4c8e64ff6a1ea3501a"
GSTREAMER_PLUGIN_URL="https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gstreamer/2a2e67491c32995a3f279ad0ecbe77abd512b42a/linuxdeploy-plugin-gstreamer.sh"
GSTREAMER_PLUGIN_SHA256="c107b49d84edbffc6ab226ed1007e0626a4f7aa2c3a36b7782bef62351d49e94"
APPIMAGE_PLUGIN_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage"
APPIMAGE_PLUGIN_SHA256="a45d3e227bc7f397e9cf6bfa4c9507494efa2293357b6e86690a3de2ca992e79"

fetch() {
  local url="$1" digest="$2" name="$3"
  local dest="$tools_dir/$name"
  curl -fsSL "$url" -o "$dest"
  echo "$digest  $dest" | sha256sum -c -
  chmod +x "$dest"
}

fetch "$LINUXDEPLOY_URL" "$LINUXDEPLOY_SHA256" linuxdeploy-x86_64.AppImage
fetch "$GTK_PLUGIN_URL" "$GTK_PLUGIN_SHA256" linuxdeploy-plugin-gtk.sh
fetch "$GSTREAMER_PLUGIN_URL" "$GSTREAMER_PLUGIN_SHA256" linuxdeploy-plugin-gstreamer.sh
fetch "$APPIMAGE_PLUGIN_URL" "$APPIMAGE_PLUGIN_SHA256" linuxdeploy-plugin-appimage.AppImage

# Replace Tauri's legacy AppRun before bundling: that binary globally prepends
# the AppDir to LD_LIBRARY_PATH and recreates #7953. linuxdeploy wraps this
# launcher with its generated GTK hook runner.
install -m 755 "$script_dir/appimage-apprun.sh" "$tools_dir/AppRun-x86_64"

# Harden the AppDir from the GTK plugin's final build-time hook. It runs after
# GTK has deployed its dependency closure and before the output plugin packages
# the generated AppRun.
install -m 755 \
  "$script_dir/finalize-complete-appimage.sh" \
  "$tools_dir/finalize-complete-appimage.sh"


# The pinned GTK plugin predates the repaired Wayland runtime boundary and
# unconditionally forces X11. Let GTK follow the session (or an explicit
# GDK_BACKEND) so the same artifact can run natively on Wayland and under X11.
sed -i '/export GDK_BACKEND=x11/d' "$tools_dir/linuxdeploy-plugin-gtk.sh"

# GIO_EXTRA_MODULES is additive, so inherited host entries must be removed.
# Pin the default module directory to the bundled modules; otherwise host proxy
# and dconf modules can be loaded into the bundled GLib process.
cat >> "$tools_dir/linuxdeploy-plugin-gtk.sh" <<'SH'
# APPIMAGE_EXTRACT_AND_RUN can pass a relative APPDIR. Canonicalize it before
# the generated hook derives WebKit helper and GTK data paths from that value.
sed -i '2i\
case "${APPDIR:-}" in\
  /*) ;;\
  *) APPDIR="$(dirname "$(realpath "$0")")" ;;\
esac\
export APPDIR
' "$HOOKFILE"


# GIO_EXTRA_MODULES is additive, so inherited host entries must be removed.
# Pin the default module directory to the bundled modules; otherwise host proxy
# and dconf modules can be loaded into the bundled GLib process.
gio_module_dir="$(find "$APPDIR"/usr/lib -type d -path '*/gio/modules' -print -quit)"
if [[ -z "$gio_module_dir" ]]; then
  echo "Complete AppImage has no bundled GIO module directory" >&2
  exit 1
fi
gio_module_rel="${gio_module_dir#"$APPDIR"/}"
cat >> "$HOOKFILE" <<EOF
unset GIO_EXTRA_MODULES
export GIO_MODULE_DIR="\$APPDIR/$gio_module_rel"
EOF

# Tauri writes .DirIcon as an absolute build-machine symlink, so it dangles on
# every user's machine and desktop integration shows no icon. Relink relative.
# Tauri points it at "$APPDIR/<product>.png", so the basename is the whole fix.
dir_icon_target="$(readlink "$APPDIR/.DirIcon" 2>/dev/null || true)"
if [[ "$dir_icon_target" == /* ]]; then
  ln -sfn "${dir_icon_target##*/}" "$APPDIR/.DirIcon"
fi

plugin_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
"$plugin_dir/finalize-complete-appimage.sh" "$APPDIR"

SH
