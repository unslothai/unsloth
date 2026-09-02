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

safe_emoji_font="/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
safe_emoji_license="/usr/share/doc/fonts-noto-color-emoji/copyright"
[[ -f "$safe_emoji_font" ]] || {
  echo "fonts-noto-color-emoji is required: $safe_emoji_font" >&2
  exit 1
}
[[ -f "$safe_emoji_license" ]] || {
  echo "fonts-noto-color-emoji license is required: $safe_emoji_license" >&2
  exit 1
}
LINUXDEPLOY_URL="https://github.com/tauri-apps/binary-releases/releases/download/linuxdeploy/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_SHA256="e762bea85c8eb0d4b3508d46e5c1f037f717d0f9303ae3b4aafc8b04991fa1ef"
GTK_PLUGIN_URL="https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gtk/b5eb8d05b4c0ed40107fe2158c5d8527f94568ef/linuxdeploy-plugin-gtk.sh"
GTK_PLUGIN_SHA256="cb379f9b0733e9ad9f8bd78f8c2fa038aef2478523bb7d4c8e64ff6a1ea3501a"
GSTREAMER_PLUGIN_URL="https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gstreamer/2a2e67491c32995a3f279ad0ecbe77abd512b42a/linuxdeploy-plugin-gstreamer.sh"
GSTREAMER_PLUGIN_SHA256="c107b49d84edbffc6ab226ed1007e0626a4f7aa2c3a36b7782bef62351d49e94"
APPIMAGE_PLUGIN_URL="https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage"
APPIMAGE_PLUGIN_SHA256="0441769ab38009504d2678c38cd7e526955388dd30a215b4a20afaa5471652f2"

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

# Replace Tauri's global LD_LIBRARY_PATH launcher to avoid #7953.
install -m 755 "$script_dir/appimage-apprun.sh" "$tools_dir/AppRun-x86_64"

# Run the finalizer after the GTK plugin deploys its dependencies.
install -m 755 \
  "$script_dir/finalize-complete-appimage.sh" \
  "$tools_dir/finalize-complete-appimage.sh"

install -m 644 "$script_dir/appimage-fonts.conf" "$tools_dir/unsloth-appimage-fonts.conf"
install -m 644 "$safe_emoji_font" "$tools_dir/UnslothSafeEmoji.ttf"
install -m 644 "$safe_emoji_license" "$tools_dir/UnslothSafeEmoji.LICENSE"

# Let GTK select X11 or Wayland instead of forcing X11.
sed -i '/export GDK_BACKEND=x11/d' "$tools_dir/linuxdeploy-plugin-gtk.sh"

# Remove host module paths from the generated AppRun hook (#7953).
cat >> "$tools_dir/linuxdeploy-plugin-gtk.sh" <<'SH'
# Canonicalize the relative APPDIR used by APPIMAGE_EXTRACT_AND_RUN.
sed -i '2i\
case "${APPDIR:-}" in\
  /*) ;;\
  *) APPDIR="$(dirname "$(realpath "$0")")" ;;\
esac\
export APPDIR
' "$HOOKFILE"

# Remove foreign-architecture GIO modules copied from multilib hosts.
while IFS= read -r -d '' gio_module; do
  machine="$(LC_ALL=C readelf -h "$gio_module" 2>/dev/null |
    sed -n 's/^[[:space:]]*Machine:[[:space:]]*//p')"
  [[ "$machine" == "Advanced Micro Devices X86-64" ]] || rm -f "$gio_module"
done < <(find "$APPDIR"/usr/lib* -path '*/gio/modules/*' -type f -print0)

# Pin GIO to the bundled modules that survived the architecture sweep.
mapfile -t gio_module_dirs < <(
  find "$APPDIR"/usr/lib* -path '*/gio/modules/*' -type f -printf '%h\n' | sort -u
)
if [[ ${#gio_module_dirs[@]} -ne 1 ]]; then
  echo "Complete AppImage needs exactly one bundled GIO module directory," \
    "found: ${gio_module_dirs[*]:-none}" >&2
  exit 1
fi
cat >> "$HOOKFILE" <<EOF
unset GIO_EXTRA_MODULES
export GIO_MODULE_DIR="\$APPDIR/${gio_module_dirs[0]#"$APPDIR"/}"
EOF

# Pin GTK_PATH so GTK_MODULES cannot load host modules into bundled GTK.
mapfile -t gtk_module_dirs < <(
  find "$APPDIR"/usr/lib* -maxdepth 2 -type d -name 'gtk-[0-9]*' | sort -u
)
if [[ ${#gtk_module_dirs[@]} -ne 1 ]]; then
  echo "Complete AppImage needs exactly one bundled GTK module directory," \
    "found: ${gtk_module_dirs[*]:-none}" >&2
  exit 1
fi
cat >> "$HOOKFILE" <<EOF
export GTK_PATH="\$APPDIR/${gtk_module_dirs[0]#"$APPDIR"/}"
EOF

# Replace Tauri's build-machine .DirIcon target with a relative link.
dir_icon_target="$(readlink "$APPDIR/.DirIcon" 2>/dev/null || true)"
if [[ "$dir_icon_target" == /* ]]; then
  ln -sfn "${dir_icon_target##*/}" "$APPDIR/.DirIcon"
fi
SH

# Append the idempotent finalizer because linuxdeploy does not guarantee plugin order.
for plugin in linuxdeploy-plugin-gtk.sh linuxdeploy-plugin-gstreamer.sh; do
  cat >> "$tools_dir/$plugin" <<'SH'

plugin_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"
"$plugin_dir/finalize-complete-appimage.sh" "$APPDIR"
SH
done
