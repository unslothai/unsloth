// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';

/**
 * Colab-style cell navigation in BOTH command and edit mode.
 *
 * ArrowDown on a cell's last line (edit) or while selected (command) moves to the
 * next cell and aligns its TOP to the viewport; ArrowUp mirrors it. JupyterLab
 * centers tall cells, dropping the view mid-output. Settings can't fix this, so
 * we listen in the CAPTURE phase, detect a cell boundary, and scroll-to-top.
 */
const cellNavPlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:cell-nav',
  description:
    'ArrowDown/ArrowUp move to the TOP of the next/previous cell (command + edit mode).',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker): void => {
    const handler = (event: KeyboardEvent): void => {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
        return;
      }
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
        return;
      }
      const panel = tracker.currentWidget;
      if (!panel || !panel.isVisible) {
        return;
      }
      if (!panel.node.contains(event.target as Node)) {
        return;
      }
      // Never hijack arrows belonging to an interactive output (ipywidgets) or a
      // form control; only the cell editor and command-mode cell nav.
      const targetEl = event.target as HTMLElement | null;
      if (targetEl) {
        if (targetEl.closest('.jp-OutputArea')) {
          return;
        }
        const tag = targetEl.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
          return;
        }
      }
      const notebook = panel.content;
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      const editing = notebook.mode === 'edit';
      if (editing) {
        const editor = notebook.activeCell?.editor;
        if (!editor) {
          return;
        }
        // While a completion popup is open the arrows belong to it; don't take
        // over even at a cell boundary (common in one-line setup cells).
        if (
          document.querySelector(
            '.jp-Completer:not(.lm-mod-hidden), .cm-tooltip-autocomplete'
          )
        ) {
          return;
        }
        const line = editor.getCursorPosition().line;
        // Only take over at the cell boundary; else let CodeMirror move the cursor.
        if (direction === 1 && line !== editor.lineCount - 1) {
          return;
        }
        if (direction === -1 && line !== 0) {
          return;
        }
      }
      const target = notebook.activeCellIndex + direction;
      if (target < 0 || target >= notebook.widgets.length) {
        return;
      }
      // We own this key: stop CodeMirror and Lumino from also handling it and
      // re-triggering the centering scroll we replace.
      event.preventDefault();
      event.stopPropagation();
      notebook.activeCellIndex = target;
      const cell = notebook.activeCell;
      const targetEditor = cell?.editor;
      if (editing && cell && targetEditor) {
        notebook.mode = 'edit';
        const lastLine = Math.max(0, targetEditor.lineCount - 1);
        targetEditor.setCursorPosition({
          line: direction === 1 ? 0 : lastLine,
          column: 0
        });
      }
      if (cell) {
        const node = cell.node;
        // Defer so this runs AFTER JupyterLab's own ensureFocus/centering scroll
        // and wins the last write. block:'start' puts the cell input at the top.
        requestAnimationFrame(() => {
          try {
            node.scrollIntoView({ block: 'start' });
          } catch {
            /* no-op */
          }
        });
      }
    };
    // Capture phase: decide before CodeMirror / Lumino consume the arrow keys.
    document.addEventListener('keydown', handler, true);
  }
};

export default cellNavPlugin;
