#!/usr/bin/env bash
# Regression test: the shim interpolates each path into a single-quoted shell
# string, so an apostrophe (/home/o'brien) closed the quote and produced a shim
# on PATH that would not parse. Generates it from the real install.sh block.
set -u
INSTALL="${1:-$(CDPATH= cd -P -- "$(dirname "$0")/../.." && pwd -P)/install.sh}"
S="$(mktemp -d)"
trap 'rm -rf "$S"' EXIT
fails=0

blk="$(awk '
  /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ { if (!seen) grab = 1 }
  /^# why: -sfn is atomic/ { if (grab) exit }
  grab { print }
  /_shim_tmp=/ { seen = 1 }
' "$INSTALL")
fi"

case "$blk" in
  *_shim_tmp*) ;;
  *) echo "FAIL: could not extract the shim block"; exit 1 ;;
esac

run_case() {
  root="$1"; label="$2"
  d="$S/case"; rm -rf "$d"; mkdir -p "$d"
  env _PORTABLE_MODE=true \
      UNSLOTH_ROOT="$root" \
      STUDIO_HOME="$root/studio" \
      VENV_DIR="$root/studio/unsloth_studio" \
      _LOCAL_BIN="$d" \
      _shim_path="$d/unsloth" \
      bash -c "substep(){ :; }; $blk" >/dev/null 2>&1

  if [ ! -f "$d/unsloth" ]; then
    echo "FAIL  no shim written: $label"; fails=$((fails+1)); return
  fi
  if ! sh -n "$d/unsloth" 2>/dev/null; then
    echo "FAIL  shim does not parse: $label"; fails=$((fails+1)); return
  fi
  sed '/^exec /d' "$d/unsloth" > "$d/vars.sh"
  got="$(sh -c ". '$d/vars.sh'; printf '%s' \"\$UNSLOTH_HOME\"" 2>/dev/null)"
  if [ "$got" != "$root" ]; then
    echo "FAIL  UNSLOTH_HOME roundtrip: $label want [$root] got [$got]"; fails=$((fails+1)); return
  fi
  got_studio="$(sh -c ". '$d/vars.sh'; printf '%s' \"\$UNSLOTH_STUDIO_HOME\"" 2>/dev/null)"
  if [ "$got_studio" != "$root/studio" ]; then
    echo "FAIL  UNSLOTH_STUDIO_HOME roundtrip: $label"; fails=$((fails+1)); return
  fi
  got_uv="$(sh -c ". '$d/vars.sh'; printf '%s' \"\$UV_CACHE_DIR\"" 2>/dev/null)"
  if [ "$got_uv" != "$root/cache/uv" ]; then
    echo "FAIL  UV_CACHE_DIR roundtrip: $label want [$root/cache/uv] got [$got_uv]"; fails=$((fails+1)); return
  fi
  if [ ! -x "$d/unsloth" ]; then
    echo "FAIL  shim not executable: $label"; fails=$((fails+1)); return
  fi
  echo "PASS  $label"
}

run_case "/plain/unsloth"                  "plain path"
run_case "/home/o'brien/unsloth"           "apostrophe (the bug)"
run_case "/home/a b/uns loth"              "spaces"
run_case "/home/quote\"dq/unsloth"         "double quote"
run_case '/home/dollar$var/unsloth'        "dollar sign"
run_case '/home/back\slash/unsloth'        "backslash"
run_case '/home/tick`cmd`/unsloth'         "backtick"
run_case '/home/semi;colon/unsloth'        "semicolon"
run_case '/home/paren(s)/unsloth'          "parentheses"
run_case '/home/star*glob/unsloth'         "glob star"
run_case "/home/ünïcodé/unsloth"           "unicode"
run_case "/home/two''quotes/unsloth"       "two apostrophes"
run_case "/home/newline_safe/unsloth"      "control"
run_case "/home/$(printf 'tab\tchar')/uns" "tab"

echo
if [ "$fails" -ne 0 ]; then echo "$fails case(s) failed"; exit 1; fi
echo "All shim path cases passed"
