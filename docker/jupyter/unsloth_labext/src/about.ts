// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
//
// "About Unsloth Docker Studio" command -> Help menu + command palette. Surfaces
// the AGPLv3 license, copyright and source/website links inside JupyterLab.

import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { Dialog, ICommandPalette, showDialog } from '@jupyterlab/apputils';
import { IMainMenu } from '@jupyterlab/mainmenu';
import { Widget } from '@lumino/widgets';
import { UNSLOTH_LOGO_DATA_URI } from './logo';
import {
  AGPL_NOTICE,
  AGPL_URL,
  APACHE_URL,
  COPYRIGHT,
  DOCS_URL,
  LICENSE_URL,
  PHRASE,
  PRODUCT,
  SHORT_LABEL,
  SOURCE_URL,
  WEBSITE_URL
} from './branding';

const COMMAND_ID = 'unsloth:about';

/**
 * Build the About dialog body from the trusted branding.ts constants only (no
 * user input, so innerHTML has no injection surface). PHRASE is stamped as a data
 * attribute so it's bundled verbatim for the integrity guard.
 */
function aboutBody(): Widget {
  const body = new Widget();
  const el = body.node;
  el.style.textAlign = 'center';
  el.style.padding = '4px 10px 10px';
  el.style.maxWidth = '430px';
  el.setAttribute('data-unsloth-attribution', PHRASE);
  // Link rows in a left-aligned inline-block centered in the dialog, so the
  // labels line up instead of each row centering independently.
  el.innerHTML = `
    <img src="${UNSLOTH_LOGO_DATA_URI}" alt="Unsloth"
         style="height:64px;width:auto;margin:2px auto 10px;display:block;" />
    <div style="font-size:16px;font-weight:700;margin-bottom:2px;">${PRODUCT}</div>
    <div style="opacity:0.8;margin-bottom:10px;">${SHORT_LABEL}</div>
    <div style="font-size:13px;line-height:1.55;margin-bottom:10px;">${AGPL_NOTICE}.</div>
    <div style="display:inline-block;text-align:left;font-size:13px;line-height:1.7;">
      <div>Source: <a href="${SOURCE_URL}" target="_blank" rel="noopener">${SOURCE_URL}</a></div>
      <div>Website: <a href="${WEBSITE_URL}" target="_blank" rel="noopener">${WEBSITE_URL}</a></div>
      <div>Unsloth Reference: <a href="${DOCS_URL}" target="_blank" rel="noopener">${DOCS_URL}</a></div>
      <div style="margin-top:8px;font-weight:600;">Licenses</div>
      <div style="margin-left:12px;">
        <div>Unsloth Studio: <a href="${AGPL_URL}" target="_blank" rel="noopener">AGPLv3</a></div>
        <div>Unsloth Core: <a href="${APACHE_URL}" target="_blank" rel="noopener">Apache 2.0</a></div>
        <div>Unsloth license: <a href="${LICENSE_URL}" target="_blank" rel="noopener">${LICENSE_URL}</a></div>
      </div>
    </div>
    <div style="font-size:12px;opacity:0.7;margin-top:12px;">${COPYRIGHT}</div>
  `;
  return body;
}

const aboutPlugin: JupyterFrontEndPlugin<void> = {
  id: 'unsloth-jupyterlab:about',
  description: 'About Unsloth Docker Studio (AGPLv3 attribution).',
  autoStart: true,
  optional: [IMainMenu, ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    mainMenu: IMainMenu | null,
    palette: ICommandPalette | null
  ): void => {
    app.commands.addCommand(COMMAND_ID, {
      label: 'About ' + PRODUCT,
      execute: () =>
        showDialog({
          title: 'About ' + PRODUCT,
          body: aboutBody(),
          buttons: [Dialog.okButton({ label: 'Close' })]
        })
    });
    if (mainMenu) {
      mainMenu.helpMenu.addGroup([{ command: COMMAND_ID }], 20);
    }
    if (palette) {
      palette.addItem({ command: COMMAND_ID, category: 'Help' });
    }
  }
};

export default aboutPlugin;
