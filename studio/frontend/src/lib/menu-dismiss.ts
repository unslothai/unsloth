/**
 * Swallow the single click that dismisses an open non-modal Radix menu.
 *
 * Why these menus are `modal={false}` at all: a modal Radix layer parks
 * `pointer-events: none` on `<body>` for as long as it is open. `pointer-events` is an
 * INHERITED property, so that one write invalidates computed style for the entire
 * mounted subtree underneath it. On a long chat thread that subtree is the thread, and
 * opening a menu turns into a full-document style recalculation whose cost scales with
 * the thread rather than with the menu.
 *
 * What that shield also did, incidentally, was absorb the click that dismisses the menu,
 * so the first click outside only ever closed it. Dropping the shield brings back a real
 * footgun: Radix's outside-pointerdown handler dismisses but never cancels the event, so
 * one click on a control next to the menu both closes the menu and fires that control.
 * In the assistant action bar the neighbours are "Refresh" and an unconfirmed
 * "Delete message", two buttons from the trigger.
 *
 * Restoring only the swallow keeps dismissal costing exactly one click, as it always
 * has, and costs no style invalidation: the listener is armed for the one click that
 * immediately follows and is disarmed either by firing or by the timeout.
 *
 * Pass as `onPointerDownOutside` on the menu content.
 */
export const swallowDismissingClick = (): void => {
  if (typeof document === "undefined") return;
  const swallow = (event: Event): void => {
    event.stopPropagation();
    event.preventDefault();
  };
  document.addEventListener("click", swallow, { capture: true, once: true });
  // A pointerdown that never becomes a click (a drag, a right click, a pointercancel)
  // would otherwise leave the swallower armed and eat an unrelated later click.
  window.setTimeout(() => {
    document.removeEventListener("click", swallow, { capture: true });
  }, 300);
};
