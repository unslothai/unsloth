// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Colab-style Ctrl/Cmd+A inside a cell output.
 *
 * Clicking a cell's output leaves the notebook in command mode, so Ctrl/Cmd+A
 * fires `notebook:select-all` (selects EVERY cell). Colab instead selects only
 * the clicked output's text; this reproduces that and stops the event so the
 * notebook-wide select-all never runs.
 *
 * Listens in the CAPTURE phase and acts only when the chord is exactly Ctrl/Cmd+A
 * (no Alt), focus is NOT in an editor/input/contenteditable, and the keystroke
 * target or last pointer-down landed in an output area. We use the last
 * pointer-down, not the text selection anchor, because a stale anchor survives a
 * click away and would hijack select-all elsewhere.
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
      // Own the chord only when in an output now: the keystroke target, else the
      // last click. Not the selection anchor -- it goes stale after clicking away
      // (see the header) and would hijack select-all elsewhere.
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
