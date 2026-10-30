// After opening /smoke-chat-background.html on the Vite server:
// playwright-cli -s=wallpaper run-code --filename=tests/studio/playwright_chat_background.js
async (page) => {
  const check=(ok,message)=>{if(!ok)throw new Error(message);};
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.addInitScript(()=>{
    window.__wallpaperWorkers=0;
    const Original=window.Worker;
    window.Worker=class extends Original{constructor(...args){super(...args);window.__wallpaperWorkers++;}};
  });
  await page.route('**/smoke-chat-background.html', async route => {
    const response = await route.fetch();
    // Vite's dev preamble needs inline scripts; workers still require self.
    await route.fulfill({response, headers:{...response.headers(), 'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; worker-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; connect-src 'self' ws:"}});
  });
  await page.reload();
  await page.getByRole('button',{name:'Reset all',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('input[type=file]')?.disabled===false);
  check(await page.getByRole('button',{name:'Remove background',exact:true}).count()===0,'no remove action without an image');
  check(await page.locator('section[data-settings-label="Background image"] [role=combobox], section[data-settings-label="Background image"] [role=slider], section[data-settings-label="Background image"] [role=switch]').count()===0,'no adjustments before upload');
  check(await page.locator('[data-wallpaper-previews] .wallpaper-preview').count()===2,'both previews visible without an image');
  check(await page.getByText('No background',{exact:true}).count()===2,'empty previews are labelled');
  check(await page.getByRole('button',{name:'Reset adjustments',exact:true}).count()===0,'no reset action without an image');
  const upload = async (invalid=false) => page.evaluate(invalid=>{
    const c=document.createElement('canvas');c.width=900;c.height=600;
    const ctx=c.getContext('2d');const g=ctx.createLinearGradient(0,0,900,600);
    g.addColorStop(0,'#e78c9c');g.addColorStop(.5,'#65d6c0');g.addColorStop(1,'#29388d');ctx.fillStyle=g;ctx.fillRect(0,0,900,600);
    for(let n=0;n<16;n++){ctx.fillStyle=n%2?'#ffffff88':'#11111188';ctx.fillRect(n*60,60,30,400);}
    const bytes=invalid?new Uint8Array([1,2,3]):Uint8Array.from(atob(c.toDataURL('image/png').split(',')[1]),v=>v.charCodeAt(0));
    const dt=new DataTransfer();dt.items.add(new File([bytes],'background.png',{type:'image/png'}));
    const input=document.querySelector('input[type=file]');input.files=dt.files;input.dispatchEvent(new Event('change',{bubbles:true}));
  },invalid);
  const settled=async(effect)=>page.waitForFunction(effect=>{
    window.__wallpaperStep=effect;const frames=document.querySelectorAll('[data-chat-content-root] .chat-wallpaper-frame');
    return frames.length===1&&frames[0].getAttribute('data-effect')===effect;
  },effect);
  await upload();await settled('None');
  check(await page.getByText('No background',{exact:true}).count()===0,'empty labels disappear after upload');
  check(await page.getByRole('button',{name:'Remove background',exact:true}).count()===1,'uploaded image can be removed');
  check(await page.getByRole('button',{name:'Reset adjustments',exact:true}).count()===0,'default adjustments need no reset');
  const canvas=page.locator('[data-wallpaper-workspace] > .chat-wallpaper');
  const crop=await canvas.boundingBox();
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  await page.locator('[data-slot=sidebar]').waitFor();
  check(JSON.stringify(crop)===JSON.stringify(await canvas.boundingBox()),'sidebar must not reframe image');
  check(await page.locator('[data-slot=sidebar]').evaluate(e=>getComputedStyle(e).backgroundColor!=='rgba(0, 0, 0, 0)'),'sidebar stays solid');
  await page.getByRole('button',{name:'Toggle page',exact:true}).click();
  await page.waitForSelector('[data-wallpaper-route="page"]');
  await canvas.waitFor({state:'detached'});
  check(await canvas.count()===0,'non-chat page has no wallpaper');
  await page.getByRole('button',{name:'Toggle page',exact:true}).click();
  await page.waitForSelector('[data-wallpaper-route="chat"]');
  await settled('None');
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  for(const effect of ['Dither','ASCII','Halftone','Scanlines','None']){
    await page.getByRole('combobox',{name:'Effect',exact:true}).click();
    await page.getByRole('option',{name:effect,exact:true}).click();
    await settled(effect);
  }
  await page.getByRole('combobox',{name:'Effect',exact:true}).click();await page.getByRole('option',{name:'ASCII',exact:true}).click();await settled('ASCII');
  const workers=await page.evaluate(()=>window.__wallpaperWorkers);
  for(const [label,size] of [['Fill','cover'],['Stretch','100% 100%'],['Original size','auto'],['Fit','contain']]) {
    await page.getByRole('combobox',{name:'Image sizing',exact:true}).click();
    await page.getByRole('option',{name:label,exact:true}).click();
    await page.waitForFunction(size=>[...document.querySelectorAll('.chat-wallpaper-artwork')].every(e=>getComputedStyle(e).backgroundSize===size),size);
  }
  await page.getByRole('slider',{name:'Zoom',exact:true}).focus();await page.keyboard.press('End');
  await page.waitForFunction(()=>getComputedStyle(document.querySelector('.chat-wallpaper-artwork')).transform==='matrix(3, 0, 0, 3, 0, 0)');
  const outline=page.getByRole('switch',{name:'Composer outline',exact:true});
  await outline.click();
  await page.waitForFunction(()=>[...document.querySelectorAll('.wallpaper-preview-composer, .unsloth-composer-surface')].every(e=>getComputedStyle(e).boxShadow==='none'));

  const fade=page.getByRole('switch',{name:'Fade to background',exact:true});
  check(await fade.getAttribute('aria-checked')==='true','fade defaults on');
  await fade.click();
  await page.waitForFunction(()=>[...document.querySelectorAll('.chat-wallpaper-image')].every(e=>getComputedStyle(e).maskImage==='none'));
  check(await page.locator('[data-chat-content-root] .chat-wallpaper-image').evaluate(e=>e.getBoundingClientRect().height===e.parentElement.getBoundingClientRect().height),'no fade fills the canvas');
  await page.getByRole('slider',{name:'Visibility',exact:true}).focus();await page.keyboard.press('End');
  await page.getByRole('slider',{name:'Image blur',exact:true}).focus();await page.keyboard.press('ArrowRight');
  await page.getByRole('button',{name:'Horizontal 100%, vertical 0%',exact:true}).click();
  await page.setViewportSize({width:1100,height:800});
  await page.getByRole('textbox',{name:'Message',exact:true}).fill('Streaming and editing do not redraw the background.');
  check(await page.evaluate(()=>window.__wallpaperWorkers)===workers,'CSS controls must not rerun effects');
  await page.reload();await settled('ASCII');
  check(await fade.getAttribute('aria-checked')==='false','fade preference persisted');
  check(await outline.getAttribute('aria-checked')==='false','outline preference persisted');
  check(await page.getByRole('slider',{name:'Zoom',exact:true}).getAttribute('aria-valuenow')==='300','zoom persisted');
  check(await page.locator('[data-chat-content-root] .chat-wallpaper-artwork').evaluate(e=>getComputedStyle(e).backgroundSize)==='contain','sizing persisted');
  check(await page.getByRole('slider',{name:'Visibility',exact:true}).getAttribute('aria-valuenow')==='100','visibility persisted');
  check(await page.getByRole('slider',{name:'Image blur',exact:true}).getAttribute('aria-valuenow')==='1','blur persisted');
  check(await page.getByRole('button',{name:'Horizontal 100%, vertical 0%',exact:true}).getAttribute('aria-pressed')==='true','position persisted');
  await upload(true);await page.getByRole('alert').waitFor();await settled('ASCII');
  await page.evaluate(async()=>{const s=window.__wallpaperSmoke;for(let i=0;i<30;i++)s.patch({visibility:i,effect:i===29?'Halftone':'Scanlines'});});
  await settled('Halftone');
  await page.waitForFunction(async()=>{const s=window.__wallpaperSmoke;return s.state().value.visibility===29;});
  await page.evaluate(()=>{window.__put=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==='backgrounds')throw new DOMException('quota','QuotaExceededError');return window.__put.apply(this,args);};});
  await upload();await page.getByRole('alert').waitFor();await settled('Halftone');
  await page.evaluate(()=>{IDBObjectStore.prototype.put=window.__put;delete window.__put;});
  await upload();await settled('None');
  check(await page.getByRole('button',{name:'Reset adjustments',exact:true}).count()===0,'reset action is absent');
  check(await fade.getAttribute('aria-checked')==='true','upload starts with fade enabled');
  check(await outline.getAttribute('aria-checked')==='true','upload starts with outline enabled');
  check(await page.getByRole('slider',{name:'Zoom',exact:true}).getAttribute('aria-valuenow')==='100','upload starts at default zoom');
  await page.getByRole('combobox',{name:'Effect',exact:true}).click();await page.getByRole('option',{name:'Halftone',exact:true}).click();await settled('Halftone');
  await page.getByRole('button',{name:'Toggle theme',exact:true}).click();await settled('Halftone');
  await page.getByRole('button',{name:'Toggle conversation',exact:true}).click();
  await page.waitForFunction(()=>getComputedStyle(document.querySelector('[data-chat-content-root] > .chat-wallpaper')).opacity==='0.4');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.waitForFunction(()=>document.querySelector('[data-chat-content-root] > .chat-wallpaper')?.getAttribute('data-reduced')==='true');
  await page.getByRole('button',{name:'Toggle route',exact:true}).click();
  check(await page.locator('[data-chat-content-root] > .chat-wallpaper').count()===0,'other route has no wallpaper');
  await page.getByRole('button',{name:'Toggle route',exact:true}).click();await settled('Halftone');
  await page.setViewportSize({width:390,height:844});
  check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile must not overflow horizontally');
  await page.screenshot({path:'.playwright-cli/wallpaper-mobile.png',fullPage:true});
  await page.setViewportSize({width:1440,height:1000});
  await page.screenshot({path:'.playwright-cli/wallpaper-conversation.png',fullPage:true});
  await page.evaluate(()=>localStorage.setItem('unsloth.browser-account.v1','account:wallpaper-test:other'));
  await page.reload();await page.waitForFunction(()=>document.querySelector('input[type=file]')?.disabled===false);
  check(await page.locator('.chat-wallpaper').count()===0,'account isolation');
  await page.evaluate(()=>localStorage.removeItem('unsloth.browser-account.v1'));
  await page.reload();await settled('Halftone');
  await page.getByRole('button',{name:'Remove background',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.chat-wallpaper'));
  check(await page.getByRole('combobox',{name:'Effect',exact:true}).count()===0,'removal hides adjustments');
  await page.reload();await page.waitForFunction(()=>document.querySelector('input[type=file]')?.disabled===false);
  check(await page.locator('.chat-wallpaper').count()===0,'removal persists');
  await upload();await settled('None');
  await page.evaluate(()=>{window.__workingWorker=window.Worker;window.Worker=function(){throw new Error('worker unavailable');};});
  await page.getByRole('combobox',{name:'Effect',exact:true}).click();await page.getByRole('option',{name:'Dither',exact:true}).click();
  await page.getByRole('alert').waitFor();await settled('None');
  await page.evaluate(()=>{window.Worker=window.__workingWorker;delete window.__workingWorker;});
  await page.getByRole('combobox',{name:'Effect',exact:true}).click();await page.getByRole('option',{name:'ASCII',exact:true}).click();await settled('ASCII');
  await page.getByRole('button',{name:'Reset all',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.chat-wallpaper'));
  return 'PASS: effects, CSS-only controls, persistence, invalid upload, quota recovery, latest-wins changes, theme, conversation, reduced motion, route lifecycle, mobile, account isolation and reset';
}
