import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio" / "frontend"
THEME_STORE = FRONTEND / "src" / "features" / "settings" / "stores" / "theme-store.ts"
INDEX_HTML = FRONTEND / "index.html"
THEME_INIT_JS = FRONTEND / "public" / "theme-init.js"
PACKAGE_LOCK = FRONTEND / "package-lock.json"
PACKAGE_JSON = FRONTEND / "package.json"
TAURI_CONF = REPO / "studio" / "src-tauri" / "tauri.conf.json"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _resolve(stored, os_dark):
    if stored == "dark":
        return "dark"
    if stored == "light":
        return "light"
    return "dark" if os_dark else "light"


def _snapshot(stored, os_dark):
    theme = stored if stored in ("light", "dark", "system") else "system"
    return f"{theme}:{_resolve(theme, os_dark)}"


def test_theme_store_snapshot_encodes_resolved():
    src = _read(THEME_STORE)
    assert "type ThemeSnapshot = `${Theme}:${ResolvedTheme}`" in src
    assert "function getSnapshot(): ThemeSnapshot {" in src
    assert "return `${theme}:${resolveTheme(theme)}`;" in src
    assert "function getServerSnapshot(): ThemeSnapshot {" in src
    assert 'return "system:light";' in src
    assert (
        'const [theme, resolved] = snapshot.split(":") as [Theme, ResolvedTheme];'
        in src
    )


def test_snapshot_changes_when_system_theme_follows_os_flip():
    assert _snapshot("system", True) == "system:dark"
    assert _snapshot("system", False) == "system:light"
    assert _snapshot("system", True) != _snapshot("system", False)


def test_snapshot_stable_when_explicit_theme_overrides_os():
    assert _snapshot("light", True) == "light:light"
    assert _snapshot("light", False) == "light:light"
    assert _snapshot("dark", True) == "dark:dark"
    assert _snapshot("dark", False) == "dark:dark"


def test_snapshot_falls_back_to_system_for_unknown_stored_value():
    assert _snapshot(None, True) == "system:dark"
    assert _snapshot("garbage", False) == "system:light"


def test_index_html_uses_external_theme_init_script_for_tauri_csp():
    html = _read(INDEX_HTML)
    assert '<script src="/theme-init.js"></script>' in html
    inline = re.search(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", html)
    assert inline is None or inline.group(1).strip() == ""


def test_theme_init_js_exists_and_handles_storage_failure_independently():
    assert THEME_INIT_JS.is_file()
    src = _read(THEME_INIT_JS)
    try_blocks = re.findall(r"\btry\s*\{", src)
    assert len(try_blocks) >= 2
    assert 'localStorage.getItem("theme")' in src
    assert 'matchMedia("(prefers-color-scheme: dark)")' in src
    assert 'classList.add("dark")' in src


def test_tauri_csp_blocks_inline_scripts_motivating_the_external_file():
    conf = json.loads(_read(TAURI_CONF))
    csp = conf["app"]["security"]["csp"]
    assert "default-src 'self'" in csp
    assert "script-src" not in csp


def test_package_lock_no_longer_references_next_themes():
    lock = json.loads(_read(PACKAGE_LOCK))
    root_deps = lock["packages"][""].get("dependencies", {})
    assert "next-themes" not in root_deps
    assert "node_modules/next-themes" not in lock["packages"]
    assert "next-themes" not in _read(PACKAGE_LOCK)


def test_package_json_has_no_next_themes_dependency():
    pkg = json.loads(_read(PACKAGE_JSON))
    assert "next-themes" not in pkg.get("dependencies", {})
    assert "next-themes" not in pkg.get("devDependencies", {})


def test_theme_store_docstring_points_to_external_bootstrap_file():
    src = _read(THEME_STORE)
    assert "public/theme-init.js" in src
    assert "inline script in index.html" not in src
