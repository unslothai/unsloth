// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Colab-style Ctrl/Cmd+A inside a cell output.
 *
 * In JupyterLab, clicking a cell's output leaves the notebook in command mode
 * (an output area is not an editor), so Ctrl/Cmd+A fires `notebook:select-all`
 * which selects EVERY cell in the notebook. On a large notebook that is both
 * surprising and laggy. Colab instead selects only the text of the output you
 * clicked. This plugin reproduces that: when the keystroke originates from
 * within an output area we select just that output's text and stop the event so
 * the notebook-wide select-all command never runs.
 *
 * We listen in the CAPTURE phase (before Lumino's command keybindings) and only
 * act when:
 *   - the chord is exactly Ctrl/Cmd+A (no Alt; Shift ignored), and
 *   - focus is NOT in a text editor / input / contenteditable (so editing a
 *     code cell with Ctrl+A still selects within that editor), and
 *   - the keystroke target OR the last pointer-down landed inside an output area.
 *
 * We deliberately do NOT use the text selection anchor to decide ownership: a
 * stale selection inside an output survives a later click onto a command-mode
 * cell or the file browser (clicking a non-text region does not always move the
 * anchor), which would make Ctrl/Cmd+A keep re-selecting that old output instead
 * of doing the normal select-all in the new context. The last pointer-down is
 * reset on every click (to null when the click is outside any output), so it
 * tracks the user's current intent; in every other case we do nothing and
 * JupyterLab keeps its default behaviour.
 */

// Output containers, widest first. `.jp-OutputArea-output` is a single output;
// `.jp-Cell-outputWrapper` is the whole output column of one cell (covers the
// case where a click lands on padding between outputs).
const OUTPUT_SELECTORS = ['.jp-OutputArea-output', '.jp-Cell-outputWrapper'];

function closestOutput(node: Node | null): HTMLElement | null {
  const el =
    node == null
      ? null
      : node.nodeType === Node.ELEMENT_NODE
        ? (node as HTMLElement)
        : node.parentElement;
  if (!el) {
    return null;
  }
  for (const sel of OUTPUT_SELECTORS) {
    const hit = el.closest(sel) as HTMLElement | null;
    if (hit) {
      return hit;
    }
  }
  return null;
}

function inEditableContext(): boolean {
  const ae = document.activeElement as HTMLElement | null;
  if (!ae) {
    return false;
  }
  if (ae.isContentEditable) {
    return true;
  }
  const tag = ae.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') {
    return true;
  }
  // CodeMirror 6 editor (cell input in edit mode).
  return !!ae.closest('.cm-editor');
}

const outputSelectPlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:output-select-all',
  description:
    'Ctrl/Cmd+A inside a cell output selects only that output, not every cell.',
  autoStart: true,
  activate: (_app: JupyterFrontEnd): void => {
    // Remember where the last pointer-down landed: a click on an image / widget
    // output may not leave a text selection inside it, so the selection anchor
    // alone is not enough to know which output the user means.
    let lastPointerOutput: HTMLElement | null = null;
    document.addEventListener(
      'pointerdown',
      (event: PointerEvent): void => {
        lastPointerOutput = closestOutput(event.target as Node | null);
      },
      true
    );

    const handler = (event: KeyboardEvent): void => {
      if (event.key !== 'a' && event.key !== 'A') {
        return;
      }
      if (!(event.ctrlKey || event.metaKey) || event.altKey) {
        return;
      }
      if (inEditableContext()) {
        return;
      }
      // Own the chord only when the user is actually in an output right now:
      // the keystroke target, else the last place they clicked. We do NOT trust
      // the text selection anchor -- it goes stale after clicking away from a
      // previously selected output (see the file header), which would otherwise
      // hijack select-all in the notebook / file browser.
      const output =
        closestOutput(event.target as Node | null) ?? lastPointerOutput;
      if (!output) {
        return;
      }
      // We own this key: prevent `notebook:select-all` (Lumino, command mode)
      // from also running and selecting the whole notebook.
      event.preventDefault();
      event.stopPropagation();
      try {
        const range = document.createRange();
        range.selectNodeContents(output);
        const sel = window.getSelection();
        if (sel) {
          sel.removeAllRanges();
          sel.addRange(range);
        }
      } catch {
        /* no-op */
      }
    };
    // Capture phase: decide before Lumino's keybindings consume Ctrl/Cmd+A.
    document.addEventListener('keydown', handler, true);
  }
};

export default outputSelectPlugin;
