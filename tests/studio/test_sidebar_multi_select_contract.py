# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for Recents multi-select bulk delete in the sidebar."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
APP_SIDEBAR = FRONTEND / "components/app-sidebar.tsx"
SELECTION_HOOK = FRONTEND / "hooks/use-sidebar-list-selection.ts"
BULK_BAR = FRONTEND / "components/sidebar-bulk-selection-bar.tsx"


def _hook() -> str:
    return SELECTION_HOOK.read_text(encoding = "utf-8")


def _sidebar() -> str:
    return APP_SIDEBAR.read_text(encoding = "utf-8")


def _bulk_bar() -> str:
    return BULK_BAR.read_text(encoding = "utf-8")


def _between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


def test_outside_pointerdown_ignores_the_portaled_confirm_dialog():
    # Clearing on the portaled dialog's buttons would drop the batch it confirms.
    block = _between(_hook(), "const isInsideSelectionBoundary", "const onPointerDown")
    assert "listRootRef.current?.contains(target)" in block
    assert '[role="dialog"], [role="alertdialog"]' in block
    assert block.index("listRootRef.current?.contains(target)") < block.index('[role="dialog"]')
    used = _between(_hook(), "const onPointerDown = (event: PointerEvent)", "clearSelection();")
    assert "isInsideSelectionBoundary(target)" in used


def test_selection_boundary_covers_the_portaled_dialog_overlay():
    # Radix renders the overlay as a sibling of the role-bearing content, so a
    # backdrop dismiss only counts as inside if the overlay matches on its own.
    block = _between(_hook(), "const isInsideSelectionBoundary", "const onPointerDown")
    assert '[data-slot$="-overlay"]' in block


def test_keyboard_activation_outside_the_list_clears_the_batch():
    # Keyboard activation fires no pointerdown; detail 0 is what marks it, and
    # pointer clicks stay on the pointerdown path.
    block = _between(_hook(), "const onClick = (event: MouseEvent)", "window.addEventListener")
    assert "if (event.detail !== 0) return;" in block
    assert "isInsideSelectionBoundary(target)" in block
    hook = _hook()
    assert 'window.addEventListener("click", onClick);' in hook
    assert 'window.removeEventListener("click", onClick);' in hook


def test_selection_drops_when_the_list_unmounts():
    # A route change unmounts the rows while the sidebar keeps the hook alive.
    block = _between(_hook(), "// The sidebar outlives the list", "// Drop stale selections")
    assert "if (selectedIds.size === 0) return;" in block
    assert "if (listRootRef.current == null) clearSelection();" in block


def test_bulk_actions_are_single_flight_and_release_on_failure():
    block = _between(_hook(), "const runBulkAction", "useEffect(")
    assert "if (bulkPendingRef.current) return false;" in block
    # The reset has to sit in finally, or a throwing batch leaves the bar dead.
    assert "} finally {" in block
    assert block.index("} finally {") < block.index("bulkPendingRef.current = false;")
    assert block.index("} finally {") < block.index("setIsBulkPending(false);")


def test_bulk_delete_loops_run_under_the_single_flight_guard():
    sidebar = _sidebar()
    chats = _between(sidebar, 'if (target.kind === "chats-bulk")', 'if (target.kind === "chat")')
    runs = _between(sidebar, 'if (target.kind === "runs-bulk")', "if (target.run.status ===")
    assert "await chatRecentsSelection.runBulkAction(async () => {" in chats
    assert "await runRecentsSelection.runBulkAction(async () => {" in runs
    # The dialog closes before the loop finishes, so the bar has to say so.
    assert "busy={chatRecentsSelection.isBulkPending}" in sidebar
    assert "busy={runRecentsSelection.isBulkPending}" in sidebar
    assert "disabled={busy}" in _bulk_bar()


def test_pointerdown_keeps_the_previous_shift_click_anchor():
    # pointerdown runs first, so reassigning the anchor there collapses the range.
    block = _between(_hook(), "const handleItemPointerDown", "const handleItemClick")
    assert "anchorIdRef.current =" not in block
    assert "anchorId: itemIds[index] ?? null" in block


def test_drag_selection_is_gated_to_mouse_pointers():
    # A touch swipe that starts on a row must still scroll the sidebar.
    hook = _hook()
    assert 'if (event.button !== 0 || event.pointerType !== "mouse") return;' in hook
    assert 'if (drag.pointerType !== "mouse") return;' in hook


def test_auto_scroll_updates_selection_and_follows_the_active_edge():
    block = _between(_hook(), "const startAutoScroll", "const updateDragSelection")
    # Rows revealed past the fold join the range, and flipping edges restarts
    # the interval instead of scrolling away.
    assert "updateDragSelectionRef.current(drag.lastClientY)" in block
    assert "autoScrollDirectionRef.current === direction" in block
    assert "autoScrollDirectionRef.current = direction" in block


def test_short_drag_is_not_undone_by_the_trailing_click():
    up = _between(_hook(), "const onPointerUp = (event: PointerEvent)", "window.addEventListener")
    assert "suppressClickRef.current = true;" in up
    assert up.index("suppressClickRef.current = true;") < up.index("dragRef.current = null;")
    click = _between(_hook(), "const handleItemClick", "const isItemSelected")
    assert "if (suppressClickRef.current) {" in click


def test_stale_selection_prune_bails_out_before_setstate():
    # itemIds is rebuilt on most renders, so an unconditional update would loop.
    block = _between(_hook(), "// Drop stale selections", "const applyRangeSelection")
    assert "if (selectedIds.size === 0) return;" in block
    assert "if (!hasStale) return;" in block


def test_bulk_bars_render_inside_their_selection_boundaries():
    sidebar = _sidebar()
    chats = _between(sidebar, "<div ref={chatRecentsListRef}>", "</SidebarGroupContent>")
    runs = _between(sidebar, "<div ref={runRecentsListRef}>", "</SidebarGroupContent>")
    assert "<SidebarBulkSelectionBar" in chats
    assert "<SidebarBulkSelectionBar" in runs


def test_bulk_chat_delete_counts_real_failures():
    sidebar = _sidebar()
    assert "): Promise<boolean> {" in _between(
        sidebar, "async function deleteChatWithCleanup", "async function handleArchiveThread"
    )
    block = _between(sidebar, 'if (target.kind === "chats-bulk")', 'if (target.kind === "chat")')
    assert "const ok = await deleteChatWithCleanup(item, { silent: true });" in block
    assert "if (!ok) failed += 1;" in block
    # Only a clean sweep clears; a partial failure keeps the rows for a retry.
    assert "if (failed === 0) {" in block
    assert "failedToDeleteSomeChats" in block


def test_selected_rows_expose_their_state_to_assistive_tech():
    sidebar = _sidebar()
    chat_row = _between(sidebar, "isActive={activeThreadId === item.id}", "className={cn(")
    run_row = _between(sidebar, "isActive={isActiveRun}", "className={cn(")
    # Only while a selection is live, else plain rows read as unpressed toggles.
    assert "aria-pressed=" in chat_row
    assert "chatRecentsSelection.isSelectionActive" in chat_row
    assert "aria-pressed=" in run_row
    assert "runRecentsSelection.isSelectionActive" in run_row


def test_bottom_fade_recomputes_when_the_bulk_bar_mounts():
    block = _between(_sidebar(), "// Recompute bottom-fade on mount", "const chatDisabled")
    assert "chatRecentsSelection.selectedCount," in block
    assert "runRecentsSelection.selectedCount," in block
