// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';

/**
 * Colab "#@title" form cells. A code cell whose first line is `#@title Some Title`
 * renders in Colab as a titled, collapsed form. JupyterLab has no equivalent, so
 * inject a clickable title bar and hide the input via a CSS class (not
 * source_hidden, so metadata is never mutated). Clicking toggles the code.
 */

const TITLE_RE = /^\s*#\s*@title\b[ \t]*(.*)$/;
const STYLE_ID = 'unsloth-colab-title-style';

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
.unsloth-title-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 4px 8px;
  /* Indent past the cell collapser + prompt gutter so the title aligns with the
     cell's input/output content column instead of the far-left edge. */
  margin: 2px 0 2px var(--jp-cell-prompt-width, 64px);
  user-select: none;
  border-radius: 4px;
  /* Heading-2-sized so a #@title form reads like a section heading (matches the
     rendered-markdown h2 scale, --jp-content-font-size4); the caret inherits
     this size so it grows too. */
  font-size: var(--jp-content-font-size4, 1.728em);
  color: var(--jp-content-font-color1, inherit);
}
.unsloth-title-bar:hover {
  background: var(--jp-layout-color2, rgba(128, 128, 128, 0.12));
}
.unsloth-title-caret {
  display: inline-block;
  width: 1em;
  line-height: 1;
  opacity: 0.8;
  transition: transform 0.12s ease;
}
.unsloth-title-bar.unsloth-collapsed .unsloth-title-caret {
  transform: rotate(-90deg);
}
.unsloth-title-text {
  font-weight: 700;
  line-height: 1.25;
}
.jp-Cell.unsloth-code-collapsed > .jp-Cell-inputWrapper {
  display: none;
}
`;
  document.head.appendChild(style);
}

function firstLineOf(cell: Cell): string {
  try {
    const raw = cell.model.toJSON().source as string | string[];
    const text = Array.isArray(raw) ? raw.join('') : String(raw || '');
    return text.split('\n', 1)[0] || '';
  } catch {
    return '';
  }
}

function applyTitle(cell: Cell): void {
  let node: HTMLElement;
  try {
    node = cell.node;
  } catch {
    return;
  }
  if (cell.model?.type !== 'code') {
    return;
  }
  const match = TITLE_RE.exec(firstLineOf(cell));
  let bar = node.querySelector(':scope > .unsloth-title-bar') as HTMLElement | null;
  if (!match) {
    if (bar) {
      bar.remove();
    }
    node.classList.remove('unsloth-titled', 'unsloth-code-collapsed');
    return;
  }
  // Drop trailing Colab form annotations, e.g. `{ display-mode: "form" }`.
  const title =
    (match[1] || '').replace(/\s*\{[^}]*\}\s*$/, '').trim() || 'Title';
  if (!bar) {
    const barEl = document.createElement('div');
    barEl.className = 'unsloth-title-bar unsloth-collapsed';
    const caret = document.createElement('span');
    caret.className = 'unsloth-title-caret';
    caret.textContent = '▾'; // down-pointing triangle
    const text = document.createElement('span');
    text.className = 'unsloth-title-text';
    barEl.appendChild(caret);
    barEl.appendChild(text);
    barEl.addEventListener('click', () => {
      const collapsed = node.classList.toggle('unsloth-code-collapsed');
      barEl.classList.toggle('unsloth-collapsed', collapsed);
    });
    node.insertBefore(barEl, node.firstChild);
    // Collapsed by default the first time we decorate this cell (Colab default).
    node.classList.add('unsloth-code-collapsed');
    bar = barEl;
  }
  const label = bar.querySelector('.unsloth-title-text') as HTMLElement | null;
  if (label) {
    label.textContent = title;
  }
  node.classList.add('unsloth-titled');
}

const colabTitlePlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:colab-title',
  description: 'Render Colab #@title code cells as collapsed, titled forms.',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker): void => {
    injectStyle();
    const decorate = (panel: NotebookPanel): void => {
      const scan = (): void => {
        panel.content.widgets.forEach(applyTitle);
      };
      panel.revealed.then(scan).catch(() => undefined);
      // Re-scan on cell add/remove/move or active-cell switch (covers editing a
      // #@title line). applyTitle never re-collapses an existing bar, so manual
      // expansions are preserved.
      const model = panel.content.model;
      if (model) {
        model.cells.changed.connect(() => window.setTimeout(scan, 0));
      }
      panel.content.activeCellChanged.connect(() => window.setTimeout(scan, 0));
    };
    tracker.widgetAdded.connect((_, panel) => decorate(panel));
    tracker.forEach(decorate);
  }
};

export default colabTitlePlugin;
