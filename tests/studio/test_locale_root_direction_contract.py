"""Regression guard for locale changes affecting the entire Unsloth layout."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOCALE_STORE = REPO / "studio/frontend/src/i18n/locale-store.ts"
MESSAGES = REPO / "studio/frontend/src/i18n/messages.ts"
SONNER = REPO / "studio/frontend/src/components/ui/sonner.tsx"


def test_locale_changes_do_not_force_document_direction():
    src = LOCALE_STORE.read_text(encoding = "utf-8")
    assert "document.documentElement.lang = locale" in src
    assert "document.documentElement.dir" not in src, (
        "locale changes must not force the root direction; html[dir] changes "
        "third-party component layout, including Sonner close-button positioning"
    )


def test_locale_metadata_does_not_advertise_unused_layout_direction():
    src = MESSAGES.read_text(encoding = "utf-8")
    locales_block = src[src.index("export const LOCALES") : src.index("export type Locale")]
    assert "dir:" not in locales_block


def test_toast_close_position_does_not_inherit_root_direction():
    src = SONNER.read_text(encoding = "utf-8")
    assert '"--toast-close-button-start": "auto"' in src
    assert '"--toast-close-button-start": "unset"' not in src
