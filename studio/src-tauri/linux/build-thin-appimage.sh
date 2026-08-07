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

stamp_appimage_bundle_type() {
  local binary_path="$1"

  python3 - "$binary_path" <<'PY'
import pathlib
import sys

binary_path = pathlib.Path(sys.argv[1])
deb_marker = b"__TAURI_BUNDLE_TYPE_VAR_DEB"
appimage_marker = b"__TAURI_BUNDLE_TYPE_VAR_APP"
binary = binary_path.read_bytes()

deb_count = binary.count(deb_marker)
appimage_count = binary.count(appimage_marker)
if deb_count != 1 or appimage_count != 0:
    sys.exit(
        f"Expected exactly one Tauri deb bundle marker and no AppImage marker in "
        f"{binary_path}; found deb={deb_count}, appimage={appimage_count}"
    )

binary_path.write_bytes(binary.replace(deb_marker, appimage_marker, 1))

stamped = binary_path.read_bytes()
if stamped.count(deb_marker) != 0 or stamped.count(appimage_marker) != 1:
    sys.exit(f"Failed to stamp the Tauri AppImage bundle marker in {binary_path}")
PY
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

# tauri-bundler stamps the executable copied into a deb so the updater selects
# the deb installer. This executable is going into an AppImage instead, so
# stamp the equal-length AppImage marker before packaging it. Without this,
# the updater downloads the AppImage and then rejects it as an invalid deb.
stamp_appimage_bundle_type "$binary_file"

ln -s usr/share/applications/Unsloth.desktop "$app_dir/Unsloth.desktop"
ln -s usr/share/icons/hicolor/128x128/apps/unsloth-studio.png "$app_dir/unsloth-studio.png"
ln -s unsloth-studio.png "$app_dir/.DirIcon"

cat > "$app_dir/AppRun" <<'EOF'
#!/bin/sh
set -eu

appdir=${APPDIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
export PATH="$appdir/usr/bin${PATH:+:$PATH}"
export XDG_DATA_DIRS="$appdir/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

binary="$appdir/usr/bin/unsloth-studio"
missing=$(
  ldd "$binary" 2>/dev/null |
    sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*$/\1/p'
)

# The tray crate loads AppIndicator with dlopen, so it does not appear in the
# executable's ldd output. Search the same host locations the loader normally
# uses, including an inherited LD_LIBRARY_PATH.
#
# libappindicator-sys prefers the versioned sonames and falls back to the
# unversioned ones, so a host that only exposes those still starts fine. This
# guard has to accept every name that loader accepts, or it refuses to launch
# on a host the tray would have worked on. The order matches the loader's.
appindicator_names="libayatana-appindicator3.so.1 libappindicator3.so.1"
appindicator_names="$appindicator_names libayatana-appindicator3.so libappindicator3.so"

appindicator_is_usable() {
  [ -r "$1" ] || return 1
  # The regular dependency guard above already treats an unavailable ldd as an
  # inconclusive check. Do the same for this dynamically loaded dependency:
  # the loader, not ldd, determines whether a readable candidate can load.
  command -v ldd >/dev/null 2>&1 || return 0
  appindicator_ldd=$(ldd "$1" 2>&1) || return 1
  ! printf '%s\n' "$appindicator_ldd" | grep -q 'not found'
}

# Every path the loader would try for one dlopen name, in its order:
# LD_LIBRARY_PATH, then the ldconfig cache, then the default directories.
appindicator_candidates() {
  library_name="$1"

  if [ "${LD_LIBRARY_PATH+x}" = x ]; then
    # The loader accepts ':' and ';' separators. An empty component is the
    # current directory. Appending a separator keeps a trailing empty
    # component visible while extracting each directory.
    library_path="${LD_LIBRARY_PATH}:"
    while [ -n "$library_path" ]; do
      library_dir=${library_path%%[:;]*}
      library_path=${library_path#*[:;]}
      [ -n "$library_dir" ] || library_dir=.
      printf '%s\n' "$library_dir/$library_name"
    done
  fi

  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk -v name="$library_name" '$1 == name { print $NF }'
  fi

  printf '%s\n' \
    /lib/"$library_name" \
    /lib/*-linux-gnu/"$library_name" \
    /lib64/"$library_name" \
    /usr/lib/"$library_name" \
    /usr/lib/*-linux-gnu/"$library_name" \
    /usr/lib64/"$library_name" \
    /usr/local/lib/"$library_name"
}

find_appindicator() {
  for library_name in $appindicator_names; do
    # The loader commits to the first file it finds for a name: if that copy
    # cannot load, the dlopen fails there instead of trying a later directory.
    # Skipping ahead would pass a host whose tray still crashes.
    first_candidate=$(
      appindicator_candidates "$library_name" |
        while IFS= read -r candidate; do
          if [ -r "$candidate" ]; then
            printf '%s\n' "$candidate"
            break
          fi
        done
    )
    [ -n "$first_candidate" ] || continue
    if appindicator_is_usable "$first_candidate"; then
      printf '%s\n' "$first_candidate"
      return 0
    fi
  done
  return 1
}

if ! find_appindicator >/dev/null; then
  if [ -n "$missing" ]; then
    missing="$missing
"
  fi
  missing="${missing}libayatana-appindicator3.so.1 or libappindicator3.so.1"
fi

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

if grep -Eq '(^|[[:space:]])(export[[:space:]]+)?LD_LIBRARY_PATH=' \
  "$verify_dir/squashfs-root/AppRun"; then
  echo "Thin AppImage must not override the host library search path." >&2
  exit 1
fi

ldd_output="$(ldd "$verify_dir/squashfs-root/usr/bin/unsloth-studio")"
if grep -q 'not found' <<< "$ldd_output"; then
  echo "The AppImage binary has unresolved dependencies on the build host." >&2
  exit 1
fi

echo "Built thin AppImage: $output_path"
