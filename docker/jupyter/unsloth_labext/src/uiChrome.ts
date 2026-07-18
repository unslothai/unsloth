// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  ILabShell,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Colab-like chrome tweaks applied image-wide.
 *
 * Hide the right activity bar (Property Inspector / Debugger tabs) by default.
 * JupyterLab has no settings key to hide a side activity bar, so hide the strip
 * with always-on CSS and collapse the right panel once on startup. Panels can
 * still be reopened from the View menu; nothing is removed, only hidden.
 */

const STYLE_ID = 'unsloth-ui-chrome-style';

function injectStyle(): void {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
/* Hide the right-hand activity bar strip (Property Inspector / Debugger tabs). */
.jp-SideBar.jp-mod-right {
  display: none !important;
}
`;
  document.head.appendChild(style);
}

const uiChromePlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:ui-chrome',
  description: 'Hide the right activity bar by default (Colab-like chrome).',
  autoStart: true,
  requires: [ILabShell],
  activate: (app: JupyterFrontEnd, shell: ILabShell): void => {
    injectStyle();
    // Collapse the right area once the layout is restored so a previously
    // expanded right panel does not linger on first paint.
    app.restored
      .then(() => {
        try {
          shell.collapseRight();
        } catch {
          /* no-op */
        }
      })
      .catch(() => undefined);
  }
};

export default uiChromePlugin;
