#!/usr/bin/env bash

set -euo pipefail

assert_thin_appdir() {
  local root="$1"
  local found=0
  local path

  while IFS= read -r -d '' path; do
    case "$(basename -- "$path")" in
      libglib-2.0.so*|libgobject-2.0.so*|libgio-2.0.so*|libgmodule-2.0.so*|\
      libgtk-3.so*|libgdk-3.so*|libgdk_pixbuf-2.0.so*|\
      libwebkit2gtk-4.1.so*|libjavascriptcoregtk-4.1.so*|libsoup-3.0.so*|\
      libcurl.so*|libcurl-gnutls.so*|libnghttp2.so*)
        echo "Thin AppImage unexpectedly contains host desktop library: $path" >&2
        found=1
        ;;
    esac
  done < <(find "$root" \( -type f -o -type l \) -print0)

  if [[ $found -ne 0 ]]; then
    return 1
  fi
}

# Keep the safety check directly executable so its behavior can be covered by
# a regression test without constructing a deb or downloading the toolchain.
if [[ $# -eq 2 && "$1" == "--verify-appdir" ]]; then
  assert_thin_appdir "$2"
  exit
fi

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 DEB APPIMAGETOOL RUNTIME OUTPUT" >&2
  echo "       $0 --verify-appdir APPDIR" >&2
  exit 2
fi

deb_path="$1"
appimagetool_path="$2"
runtime_path="$3"
output_path="$4"

for input_path in "$deb_path" "$appimagetool_path" "$runtime_path"; do
  if [[ ! -f "$input_path" ]]; then
    echo "Required AppImage input does not exist: $input_path" >&2
    exit 1
  fi
done

if [[ "$(dpkg-deb --field "$deb_path" Architecture)" != "amd64" ]]; then
  echo "The Linux AppImage must be built from an amd64 deb." >&2
  exit 1
fi

work_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/unsloth-thin-appimage.XXXXXX")"
trap 'rm -rf -- "$work_root"' EXIT
app_dir="$work_root/AppDir"
verify_dir="$work_root/verify"
output_dir="$(dirname -- "$output_path")"
mkdir -p "$app_dir" "$verify_dir" "$output_dir"
output_dir="$(CDPATH= cd -- "$output_dir" && pwd -P)"
output_path="$output_dir/$(basename -- "$output_path")"

# The deb is already a working host-integrated package. Reusing its payload
# avoids a partial GTK/WebKit dependency closure, which can mix an
# old bundled GLib with newer host GIO modules and network libraries.
dpkg-deb --extract "$deb_path" "$app_dir"

desktop_file="$app_dir/usr/share/applications/Unsloth.desktop"
icon_file="$app_dir/usr/share/icons/hicolor/128x128/apps/unsloth-studio.png"
binary_file="$app_dir/usr/bin/unsloth-studio"
for required_path in "$desktop_file" "$icon_file" "$binary_file"; do
  if [[ ! -f "$required_path" ]]; then
    echo "The deb is missing an AppImage input: $required_path" >&2
    exit 1
  fi
done

ln -s usr/share/applications/Unsloth.desktop "$app_dir/Unsloth.desktop"
ln -s usr/share/icons/hicolor/128x128/apps/unsloth-studio.png "$app_dir/unsloth-studio.png"
ln -s unsloth-studio.png "$app_dir/.DirIcon"

cat > "$app_dir/AppRun" <<'EOF'
#!/bin/sh
set -eu

appdir=${APPDIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
export PATH="$appdir/usr/bin${PATH:+:$PATH}"
export XDG_DATA_DIRS="$appdir/usr/share${XDG_DATA_DIRS:+:$XDG_DATA_DIRS}"

binary="$appdir/usr/bin/unsloth-studio"
missing=$(
  ldd "$binary" 2>/dev/null |
    sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*$/\1/p'
)
if [ -n "$missing" ]; then
  message="Unsloth cannot start because required Linux libraries are missing:

$missing

On Ubuntu or Linux Mint, install them with:
sudo apt install libayatana-appindicator3-1 libwebkit2gtk-4.1-0 libgtk-3-0"
  printf '%s\n' "$message" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="Unsloth cannot start" --text="$message" >/dev/null 2>&1 || true
  elif command -v xmessage >/dev/null 2>&1; then
    xmessage -center "$message" >/dev/null 2>&1 || true
  fi
  exit 127
fi

exec "$binary" "$@"
EOF
chmod +x "$app_dir/AppRun"

assert_thin_appdir "$app_dir"

ARCH=x86_64 "$appimagetool_path" --appimage-extract-and-run \
  --no-appstream \
  --runtime-file "$runtime_path" \
  "$app_dir" "$output_path"

if [[ ! -x "$output_path" ]]; then
  echo "appimagetool did not create an executable AppImage: $output_path" >&2
  exit 1
fi

(
  cd "$verify_dir"
  "$output_path" --appimage-extract >/dev/null
)
assert_thin_appdir "$verify_dir/squashfs-root"

if grep -q 'LD_LIBRARY_PATH' "$verify_dir/squashfs-root/AppRun"; then
  echo "Thin AppImage must not override the host library search path." >&2
  exit 1
fi

ldd_output="$(ldd "$verify_dir/squashfs-root/usr/bin/unsloth-studio")"
if grep -q 'not found' <<< "$ldd_output"; then
  echo "The AppImage binary has unresolved dependencies on the build host." >&2
  exit 1
fi

echo "Built thin AppImage: $output_path"
