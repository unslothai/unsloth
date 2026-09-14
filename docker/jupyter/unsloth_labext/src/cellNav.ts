// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { CodeMirrorEditor } from '@jupyterlab/codemirror';
import { INotebookTracker } from '@jupyterlab/notebook';

/**
 * Colab-style cell navigation: at a cell boundary, move to the next cell and align
 * its TOP to the viewport. JupyterLab centers tall cells instead, dropping the view
 * mid-output, and no setting changes that.
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
        if (
          document.querySelector(
            '.jp-Completer:not(.lm-mod-hidden), .cm-tooltip-autocomplete'
          )
        ) {
          return;
        }
        // `lineCount` counts LOGICAL lines, but JupyterLab wraps markdown and raw
        // editors by default, so one logical line can own several visual rows and a
        // logical test makes them unreachable: every arrow leaves the cell.
        const view = editor instanceof CodeMirrorEditor ? editor.editor : null;
        if (view) {
          const range = view.state.selection.main;
          const moved = view.moveVertically(range, direction === 1);
          const from = view.coordsAtPos(range.head);
          const to =
            moved.head === range.head ? from : view.coordsAtPos(moved.head);
          // moveVertically returns an unchanged head only at the document edge
          if (from && to && Math.abs(to.top - from.top) > 1) {
            return;
          }
        } else {
          const line = editor.getCursorPosition().line;
          if (direction === 1 && line !== editor.lineCount - 1) {
            return;
          }
          if (direction === -1 && line !== 0) {
            return;
          }
        }
      }
      const target = notebook.activeCellIndex + direction;
      if (target < 0 || target >= notebook.widgets.length) {
        return;
      }
      // stop Lumino re-triggering the centering scroll we replace
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
        // defer, so this wins over JupyterLab's own centering scroll
        requestAnimationFrame(() => {
          try {
            node.scrollIntoView({ block: 'start' });
          } catch {
            /* no-op */
          }
        });
      }
    };
    document.addEventListener('keydown', handler, true);
  }
};

export default cellNavPlugin;
