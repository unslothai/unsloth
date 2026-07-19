"""Compact viewport contracts for nested dropdown menus."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DROPDOWN_MENU = REPO / "studio/frontend/src/components/ui/dropdown-menu.tsx"
FRONTEND_SRC = REPO / "studio/frontend/src"


def test_shared_submenu_uses_its_layout_width_on_mobile():
    source = DROPDOWN_MENU.read_text(encoding = "utf-8")
    assert 'import { useIsMobile } from "@/hooks/use-mobile";' in source
    assert "element.offsetWidth" in source
    assert "element.getBoundingClientRect().width" not in source
    assert "new ResizeObserver(updateContentWidth)" in source
    assert "isMobile && contentWidth > 0 ? -contentWidth : sideOffset" in source
    assert "sideOffset={compactSideOffset}" in source
    assert 'isMobile && contentWidth === 0 ? "hidden"' in source
    assert "-248" not in source


def test_shared_submenu_never_exceeds_the_compact_viewport():
    source = DROPDOWN_MENU.read_text(encoding = "utf-8")
    assert "max-w-[calc(100vw-2rem)]" in source


def test_consumers_do_not_duplicate_compact_offset_logic():
    for path in FRONTEND_SRC.rglob("*.tsx"):
        if path == DROPDOWN_MENU:
            continue
        source = path.read_text(encoding = "utf-8")
        assert "compactSubmenuOffset" not in source, path


def test_all_submenu_consumers_use_the_shared_primitive():
    for path in FRONTEND_SRC.rglob("*.tsx"):
        if path == DROPDOWN_MENU:
            continue
        source = path.read_text(encoding = "utf-8")
        assert "DropdownMenuPrimitive.SubContent" not in source, path
