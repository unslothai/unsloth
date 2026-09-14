// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  ILabShell,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

/**
 * Hide the right activity bar by default. There is no settings key for it, so the
 * strip is hidden with CSS and the panel collapsed once on startup.
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
