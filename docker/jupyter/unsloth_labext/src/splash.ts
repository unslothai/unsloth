// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
//
// Replace the JupyterLab loading splash with a spinning Unsloth logo. Provides
// the core ISplashScreen token; the stock splash is disabled + locked at build,
// so this is the only provider. Animation honors prefers-reduced-motion.

import { JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ISplashScreen } from '@jupyterlab/apputils';
import { DisposableDelegate, IDisposable } from '@lumino/disposable';
import { UNSLOTH_LOGO_DATA_URI } from './logo';
import { SPLASH_LABEL } from './branding';

const STYLE_ID = 'unsloth-splash-style';
const SPLASH_ID = 'unsloth-splash';

function ensureStyle(): void {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
#${SPLASH_ID} {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--jp-layout-color0, hsl(70, 8%, 12%));
}
#${SPLASH_ID} img {
  height: 72px;
  width: 72px;
  animation: unsloth-splash-spin 1.2s linear infinite;
}
#${SPLASH_ID} .unsloth-splash-label {
  margin-top: 14px;
  font-size: 13px;
  opacity: 0.7;
  font-family: sans-serif;
  color: var(--jp-ui-font-color1, hsl(60, 30%, 92%));
}
@keyframes unsloth-splash-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
  #${SPLASH_ID} img { animation: none; }
}
`;
  document.head.appendChild(style);
}

const splashPlugin: JupyterFrontEndPlugin<ISplashScreen> = {
  id: 'unsloth-jupyterlab:splash',
  description: 'Unsloth spinning-logo loading splash.',
  autoStart: true,
  provides: ISplashScreen,
  activate: (): ISplashScreen => {
    return {
      show: (): IDisposable => {
        ensureStyle();
        const overlay = document.createElement('div');
        overlay.id = SPLASH_ID;

        const img = document.createElement('img');
        img.src = UNSLOTH_LOGO_DATA_URI;
        img.alt = 'Unsloth';
        overlay.appendChild(img);

        const label = document.createElement('div');
        label.className = 'unsloth-splash-label';
        label.textContent = SPLASH_LABEL;
        overlay.appendChild(label);

        document.body.appendChild(overlay);
        return new DisposableDelegate(() => {
          overlay.remove();
        });
      }
    };
  }
};

export default splashPlugin;
