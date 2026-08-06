


/** What a tooltip renders, given everything that can hold it open or shut.
 *
 * Radix is always given a boolean so it keeps no open state of its own that
 * could survive a modal and resurface after it closes.
 */
export function resolveTooltipOpen(state: {
  /** A modal owns the screen and this tooltip's trigger is not inside it. */
  blocked: boolean;
  /** The consumer's `open`, or undefined when this tooltip is uncontrolled. */
  controlledOpen?: boolean;
  /** A modal blocked it and the owner has not said false since. */
  dismissedUntilOwnerResets: boolean;
  /** Radix's hover/focus state, tracked here rather than left inside it. */
  hoverOpen: boolean;
  /** Pinned by a tap, which has no hover to close it. */
  clickOpen: boolean;
}): boolean {
  if (state.blocked) return false;
  if (state.controlledOpen !== undefined) {
    // An owner with no onOpenChange (panel-resize-handle) never learns its
    // trigger lost the pointer, so its `open` is still true when the modal
    // closes. Honour it again only after it has gone false once.
    return state.controlledOpen && !state.dismissedUntilOwnerResets;
  }
  return state.hoverOpen || state.clickOpen;
}
