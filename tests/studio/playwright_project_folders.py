# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drive real project dialogs with stubbed native IPC and HTTP.

Start the frontend with npm run dev, then run this driver with --base-url
pointing to it. --before-url optionally captures the same dialog from a second
frontend. No model, real native picker, or host filesystem grant is involved.
"""
import asyncio
import argparse
import json
from pathlib import Path
from playwright.async_api import async_playwright, expect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:5173')
parser.add_argument('--before-url')
parser.add_argument('--out', type=Path, default=Path('work/project-folders-review'))
ARGS = parser.parse_args()
OUT = ARGS.out.resolve()

async def scene(browser, label, base_url, mode, *, exercise=False, reject=False,
                cancel=False, reset=False, pending=False, narrow=False):
    context = await browser.new_context(
        viewport={"width":390,"height":750} if narrow else {"width":1100,"height":950},
        device_scale_factor=1,
    )
    await context.add_init_script("""
      localStorage.setItem('unsloth_auth_token', 'review.'+btoa(JSON.stringify({sub:'unsloth', role:'owner'}))+'.review');
      window.__picks = 0;
    """)
    await context.add_init_script(
        f"window.__cancelPick={str(cancel).lower()}; window.__delayPick={str(pending).lower()};"
    )
    if mode == 'guest':
        await context.add_init_script("""
          localStorage.setItem('unsloth_auth_token', 'review.'+btoa(JSON.stringify({sub:'member',role:'member'}))+'.review');
        """)
    if mode != 'browser':
        await context.add_init_script("""
          window.__TAURI_INTERNALS__ = {
            metadata: {currentWindow:{label:'main'}, currentWebview:{label:'main'}},
            transformCallback: () => 1, unregisterCallback: () => {},
            invoke: async (command) => {
              if(command === 'native_path_leases_usable') return true;
              if(command === 'pick_native_document_folder') {
                window.__picks++;
                if(window.__delayPick) return new Promise(resolve => {window.__resolvePick=resolve;});
                if(window.__cancelPick) return null;
                return {token:'native-grant-'+window.__picks, displayName:'Existing project', expiresAtMs:Date.now()+120000};
              }
              if(command.endsWith('scale_factor')) return 1;
              if(command.endsWith('listen')) return 1;
              return null;
            }
          };
        """)
    requests=[]
    folder_requests=[]
    stored={}
    failures = int(reject)
    async def api(route):
        nonlocal failures
        req=route.request
        path=req.url.split('/api/',1)[-1]
        if path == 'health':
            return await route.fulfill(json={"native_path_leases_supported":mode in ('desktop','guest')})
        if path == 'chat/projects' and req.method == 'POST':
            data=req.post_data_json
            requests.append(data)
            if failures:
                failures -= 1
                return await route.fulfill(status=400,json={"detail":"Folder selection expired"})
            stored.update({**data,"workingFolder":"/review/existing-project" if data.get('nativePathLease') else None})
            return await route.fulfill(json=stored)
        if path.endswith('/working-folder'):
            folder_requests.append({"method":req.method,"body":req.post_data_json if req.method=='PUT' else None})
            if await page.evaluate('Boolean(window.__rejectUpdate)'):
                await page.evaluate('window.__rejectUpdate=false')
                return await route.fulfill(status=400,json={"detail":"Folder selection expired"})
            if await page.evaluate('Boolean(window.__delayUpdate)'):
                await page.wait_for_function('!window.__delayUpdate')
            stored['workingFolder']='/review/alternate-project' if req.method=='PUT' else None
            return await route.fulfill(json=stored)
        if path.startswith('chat/projects/') and req.method=='GET':
            return await route.fulfill(json=stored)
        return await route.fulfill(json={})
    await context.route(f'{base_url}/api/**', api)
    page=await context.new_page()
    errors=[]
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('pageerror', lambda error: print('PAGE ERROR', label, str(error), flush=True))
    print('SCENE', label, flush=True)
    await page.goto(f'{base_url}/smoke-project-folders.html')
    dialog=page.get_by_role('dialog')
    await expect(dialog).to_be_visible(timeout=90000)
    await page.get_by_label('Project name', exact=True).fill('My project')
    observations = {
        "workingFolderControls": await dialog.get_by_role('button', name='Choose folder…', exact=True).count(),
        "sourceFolderLabels": await dialog.get_by_text('Source folders', exact=True).count(),
        "technicalBackendCopy": 'managed desktop backend' in await dialog.inner_text(),
    }
    await page.evaluate('document.fonts.ready')
    await page.screenshot(path=str(OUT/f'{label}.png'), animations='disabled')
    await dialog.screenshot(path=str(OUT/f'{label}-dialog.png'), animations='disabled')
    if exercise:
        if mode=='desktop':
            choose=dialog.get_by_role('button',name='Choose folder…',exact=True)
            await expect(choose).to_be_enabled()
            await choose.click()
            if pending:
                await expect(dialog.get_by_role('button',name='Choosing folder…',exact=True)).to_be_disabled()
                await page.keyboard.press('Escape')
                await expect(dialog).to_be_visible()
                await page.evaluate('window.__resolvePick(null)')
            if cancel or pending:
                await expect(dialog.get_by_text('Studio-managed folder',exact=True)).to_be_visible()
                await expect(choose).to_be_enabled()
            else:
                await expect(dialog.get_by_text('Existing project',exact=True)).to_be_visible()
            if reset:
                await dialog.get_by_role('button',name='Use managed folder',exact=True).click()
                await expect(dialog.get_by_text('Studio-managed folder',exact=True)).to_be_visible()
            if narrow:
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            await page.screenshot(path=str(OUT/f'{label}-selected.png'), animations='disabled')
            await dialog.screenshot(path=str(OUT/f'{label}-selected-dialog.png'), animations='disabled')
            await dialog.get_by_role('button',name='Create project',exact=True).click()
            if reject:
                await expect(dialog.get_by_role('alert')).to_contain_text('Choose the folder again')
                assert len(requests)==1
                await expect(dialog.get_by_role('button',name='Create project',exact=True)).to_be_disabled()
                await page.screenshot(path=str(OUT/f'{label}-expired.png'), animations='disabled')
                await dialog.get_by_role('button',name='Change…',exact=True).click()
                await dialog.get_by_role('button',name='Create project',exact=True).click()
            await expect(page.get_by_test_id('created')).not_to_be_empty()
            if cancel or pending or reset:
                assert 'nativePathLease' not in requests[-1]
            else:
                assert requests[-1]['nativePathLease'].startswith('native-grant-')
            assert requests[-1]['name']=='My project'
            if label=='after-desktop':
                await page.get_by_role('button',name='Edit project',exact=True).click()
                await expect(dialog.get_by_text('/review/existing-project',exact=True)).to_be_visible()
                await expect(dialog.get_by_text('Where code and file tools run for every chat in this project.',exact=True)).to_be_visible()
                await page.evaluate('window.__cancelPick=true')
                await dialog.get_by_role('button',name='Change…',exact=True).click()
                await expect(dialog.get_by_role('button',name='Change…',exact=True)).to_be_enabled()
                assert not folder_requests
                await page.evaluate('window.__cancelPick=false; window.__rejectUpdate=true')
                await dialog.get_by_role('button',name='Change…',exact=True).click()
                await expect(page.get_by_text('Could not change the working folder',exact=True)).to_be_visible()
                await expect(dialog.get_by_text('/review/existing-project',exact=True)).to_be_visible()
                await page.evaluate('window.__delayUpdate=true')
                await dialog.get_by_role('button',name='Change…',exact=True).click()
                await expect(dialog.get_by_role('button',name='Save',exact=True)).to_be_disabled()
                await expect(dialog.get_by_role('button',name='Cancel',exact=True)).to_be_disabled()
                await expect(dialog.get_by_role('button',name='Delete project',exact=True)).to_be_disabled()
                await page.keyboard.press('Escape')
                await expect(dialog).to_be_visible()
                await page.evaluate('window.__delayUpdate=false')
                await expect(dialog.get_by_text('/review/alternate-project',exact=True)).to_be_visible()
                await dialog.screenshot(path=str(OUT/'after-edit-selected-dialog.png'),animations='disabled')
                await dialog.get_by_role('button',name='Cancel',exact=True).click()
                await page.get_by_role('button',name='Edit project',exact=True).click()
                await expect(dialog.get_by_text('/review/alternate-project',exact=True)).to_be_visible()
                await dialog.get_by_role('button',name='Use managed folder',exact=True).click()
                await expect(dialog.get_by_text('Studio-managed folder',exact=True)).to_be_visible()
                await dialog.get_by_role('button',name='Cancel',exact=True).click()
                await page.get_by_role('button',name='Edit project',exact=True).click()
                await expect(dialog.get_by_text('Studio-managed folder',exact=True)).to_be_visible()
                assert [r['method'] for r in folder_requests]==['PUT','PUT','DELETE']
        else:
            if mode in ('unsupported','guest'):
                await expect(dialog.get_by_role('button',name='Choose folder…',exact=True)).to_be_disabled()
            await dialog.get_by_role('button',name='Create project',exact=True).click()
            await expect(page.get_by_test_id('created')).not_to_be_empty()
            assert 'nativePathLease' not in requests[-1]
    assert not errors, errors
    facts={"mode":mode,"requests":requests,"folderRequests":folder_requests,"pageErrors":errors,"rejectedGrant":reject,
           "cancelledPicker":cancel,"resetToDefault":reset,"pendingPicker":pending,"narrowViewport":narrow,
           **observations}
    await context.close()
    return facts

async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True)
        facts={}
        if ARGS.before_url:
            facts['before-browser']=await scene(browser,'before-browser',ARGS.before_url,'browser')
        facts['after-browser']=await scene(browser,'after-browser',ARGS.base_url,'browser',exercise=True)
        facts['after-desktop-unavailable']=await scene(browser,'after-desktop-unavailable',ARGS.base_url,'unsupported',exercise=True)
        facts['after-desktop']=await scene(browser,'after-desktop',ARGS.base_url,'desktop',exercise=True)
        facts['after-retry']=await scene(browser,'after-retry',ARGS.base_url,'desktop',exercise=True,reject=True)
        facts['after-picker-cancel']=await scene(browser,'after-picker-cancel',ARGS.base_url,'desktop',exercise=True,cancel=True)
        facts['after-use-default']=await scene(browser,'after-use-default',ARGS.base_url,'desktop',exercise=True,reset=True)
        facts['after-picker-pending']=await scene(browser,'after-picker-pending',ARGS.base_url,'desktop',exercise=True,pending=True)
        facts['after-managed-account']=await scene(browser,'after-managed-account',ARGS.base_url,'guest',exercise=True)
        facts['after-narrow']=await scene(browser,'after-narrow',ARGS.base_url,'desktop',exercise=True,narrow=True)
        await browser.close()
    (OUT/'facts.json').write_text(json.dumps(facts,indent=2))
    print(json.dumps({"scenes":len(facts),"output":str(OUT)},indent=2))

asyncio.run(main())
