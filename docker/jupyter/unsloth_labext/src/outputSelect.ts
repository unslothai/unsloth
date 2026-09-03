// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Colab-style Ctrl/Cmd+A inside a cell output: clicking an output leaves the
 * notebook in command mode, so the chord otherwise fires `notebook:select-all`.
 * Keyed off the target or last pointer-down, never the stale selection anchor.
 */

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
  return !!ae.closest('.cm-editor');
}

const outputSelectPlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:output-select-all',
  description:
    'Ctrl/Cmd+A inside a cell output selects only that output, not every cell.',
  autoStart: true,
  activate: (_app: JupyterFrontEnd): void => {
    // a click on an image/widget output leaves no text selection
    let lastPointerOutput: HTMLElement | null = null;
    // ...but only while it is still in the document AND in the ACTIVE cell: J/K
    // navigation fires no pointer event, so a stale value selects an earlier cell's
    // output, and a re-executed cell leaves a detached range that selects nothing
    const rememberedOutput = (): HTMLElement | null => {
      const output = lastPointerOutput;
      if (!output || !output.isConnected) {
        return null;
      }
      const cell = output.closest('.jp-Cell');
      return cell && cell.classList.contains('jp-mod-active') ? output : null;
    };
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
      const output =
        closestOutput(event.target as Node | null) ?? rememberedOutput();
      if (!output) {
        return;
      }
      // prevent Lumino's `notebook:select-all` from also running
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
    document.addEventListener('keydown', handler, true);
  }
};

export default outputSelectPlugin;
