"""Regression guard for locale changes affecting the entire Unsloth layout."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOCALE_STORE = REPO / "studio/frontend/src/i18n/locale-store.ts"
I18N = REPO / "studio/frontend/src/i18n"
SONNER = REPO / "studio/frontend/src/components/ui/sonner.tsx"


def test_locale_changes_do_not_force_document_direction():
    src = LOCALE_STORE.read_text(encoding = "utf-8")
    assert "document.documentElement.lang = locale" in src
    assert "document.documentElement.dir" not in src, (
        "locale changes must not force the root direction; html[dir] changes "
        "third-party component layout, including Sonner close-button positioning"
    )


def _locales_block():
    """The `LOCALES` table's source, from whichever i18n module defines it (messages.ts re-exports it)."""
    marker = "export const LOCALES = {"
    owners = [
        path for path in sorted(I18N.rglob("*.ts")) if marker in path.read_text(encoding = "utf-8")
    ]
    assert len(owners) == 1, f"expected one module defining LOCALES under {I18N}, found {owners}"
    src = owners[0].read_text(encoding = "utf-8")
    start = src.index(marker)
    end = src.find("} as const;", start)
    assert end != -1, f"LOCALES in {owners[0]} no longer ends with `}} as const;`"
    return src[start:end]


def test_locale_metadata_does_not_advertise_unused_layout_direction():
    block = _locales_block()
    assert '"en"' in block or "en:" in block, "the LOCALES block was not read"
    assert "dir:" not in block


def test_toast_close_position_does_not_inherit_root_direction():
    src = SONNER.read_text(encoding = "utf-8")
    assert '"--toast-close-button-start": "auto"' in src
    assert '"--toast-close-button-start": "unset"' not in src
