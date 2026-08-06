


/**
 * Whether a tooltip trigger sits below the active modal rather than inside it.
 *
 * Radix's DismissableLayer writes `pointer-events` inline: `none` on the body
 * while a modal is up, `auto` on the active layer, `none` on the layers under
 * it. Walking the ancestors for the nearest of those answers which layer owns
 * the trigger.
 *
 * The trigger's own style is deliberately skipped. A trigger can be authored
 * `pointer-events: none` and still belong to the modal, which is exactly what
 * the hint anchors inside the MCP dropdown rows do.
 *
 * Read the trigger, never the tooltip content: layers rank by mount order, so
 * content opened after the modal reads `auto` wherever its trigger sits.
 */
export function isBlockedByActiveModal(element: HTMLElement): boolean {
  for (let node = element.parentElement; node; node = node.parentElement) {
    const pointerEvents = node.style?.pointerEvents;
    if (pointerEvents === "auto") return false;
    if (pointerEvents === "none") return true;
  }
  return false;
}
