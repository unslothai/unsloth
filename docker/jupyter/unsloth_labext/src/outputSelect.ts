// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Colab-style Ctrl/Cmd+A inside a cell output.
 *
 * Clicking an output leaves the notebook in command mode, so Ctrl/Cmd+A fires
 * `notebook:select-all` (every cell). Colab selects only the clicked output's
 * text; reproduce that and stop the event. Listens in the CAPTURE phase, acts
 * only on exactly Ctrl/Cmd+A (no Alt) outside an editor/input, keyed off the
 * target or last pointer-down (not the stale selection anchor).
 */

// Output containers, widest first: a single output, then the whole output column
// (covers a click on padding between outputs).
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
    // Remember the last pointer-down: a click on an image/widget output leaves no
    // text selection, so the anchor alone can't tell which output is meant.
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
      // Own the chord only when in an output: the target, else the last click
      // (not the stale selection anchor; see the header).
      const output =
        closestOutput(event.target as Node | null) ?? lastPointerOutput;
      if (!output) {
        return;
      }
      // We own this key: prevent Lumino's `notebook:select-all` from also running.
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
