"""Regression guard for locale changes affecting the entire Studio layout."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LOCALE_STORE = REPO / "studio/frontend/src/i18n/locale-store.ts"
MESSAGES = REPO / "studio/frontend/src/i18n/messages.ts"


def test_locale_changes_do_not_force_document_direction():
    src = LOCALE_STORE.read_text()
    assert "document.documentElement.lang = locale" in src
    assert "document.documentElement.dir" not in src, (
        "locale changes must not force the root direction; html[dir] changes "
        "third-party component layout, including Sonner close-button positioning"
    )


def test_locale_metadata_does_not_advertise_unused_layout_direction():
    src = MESSAGES.read_text()
    locales_block = src[src.index("export const LOCALES") : src.index("export type Locale")]
    assert "dir:" not in locales_block
