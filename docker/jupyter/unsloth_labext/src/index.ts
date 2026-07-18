// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

import {
  ILabShell,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { IThemeManager } from '@jupyterlab/apputils';
import { Widget } from '@lumino/widgets';
import { UNSLOTH_LOGO_DATA_URI } from './logo';
import aboutPlugin from './about';
import cellNavPlugin from './cellNav';
import colabTitlePlugin from './colabTitle';
import outputSelectPlugin from './outputSelect';
import splashPlugin from './splash';
import uiChromePlugin from './uiChrome';

/**
 * The "Unsloth Dark" theme: JupyterLab Dark repainted with the Sublime/Colab
 * Monokai palette (see style/variables.css). Registered as a named theme so it
 * appears in Settings > Theme and works with the adaptive (system) light/dark
 * switch configured in overrides.json.
 */
const themePlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:theme',
  description: 'Unsloth Dark (Monokai) theme.',
  autoStart: true,
  requires: [IThemeManager],
  activate: (app: JupyterFrontEnd, manager: IThemeManager): void => {
    const style = 'unsloth-jupyterlab/index.css';
    manager.register({
      name: 'Unsloth Dark',
      isLight: false,
      themeScrollbars: true,
      load: () => manager.loadCSS(style),
      unload: () => Promise.resolve(undefined)
    });
  }
};

/**
 * Replace the top-left Jupyter logo with the Unsloth logo. The stock logo plugin
 * is disabled + locked at build time, so this is the only logo widget. Rendered
 * as an <img> with inline styles (not a LabIcon/CSS) so branding shows identically
 * in any theme (the theme CSS loads only while Unsloth Dark is selected).
 */
const logoPlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:logo',
  description: 'Replace the top-left Jupyter logo with the Unsloth logo.',
  autoStart: true,
  requires: [ILabShell],
  activate: (app: JupyterFrontEnd, shell: ILabShell): void => {
    const logo = new Widget();
    const img = document.createElement('img');
    img.src = UNSLOTH_LOGO_DATA_URI;
    img.alt = 'Unsloth';
    img.style.height = '24px';
    img.style.width = 'auto';
    img.style.margin = '1px 6px 1px 8px';
    img.style.display = 'block';
    logo.node.appendChild(img);
    logo.node.style.display = 'flex';
    logo.node.style.alignItems = 'center';
    logo.id = 'jp-MainLogo';
    shell.add(logo, 'top', { rank: 0 });
  }
};

export default [
  themePlugin,
  cellNavPlugin,
  logoPlugin,
  colabTitlePlugin,
  outputSelectPlugin,
  uiChromePlugin,
  aboutPlugin,
  splashPlugin
];
