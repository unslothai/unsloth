// Run against the Vite server with:
// playwright-cli -s=wallpaper run-code --filename=tests/studio/playwright_chat_background.js
// The session must have opened /smoke-chat-background.html first.
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
  const canvas=page.locator('[data-wallpaper-workspace] > .chat-wallpaper');
  const crop=await canvas.boundingBox();
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  await page.locator('[data-slot=sidebar]').waitFor();
  check(JSON.stringify(crop)===JSON.stringify(await canvas.boundingBox()),'sidebar must not reframe image');
  check(await page.locator('[data-slot=sidebar]').evaluate(e=>getComputedStyle(e).backgroundColor!=='rgba(0, 0, 0, 0)'),'sidebar stays solid');
  await page.getByRole('button',{name:'Toggle page',exact:true}).click();
  await page.waitForSelector('[data-wallpaper-route="page"]');
  check(await canvas.evaluate(e=>getComputedStyle(e).opacity)==='1','page wallpaper stays visible');
  await page.getByRole('button',{name:'Toggle page',exact:true}).click();
  await page.waitForSelector('[data-wallpaper-route="chat"]');
  await page.getByRole('button',{name:'Toggle sidebar',exact:true}).click();
  for(const effect of ['Dither','ASCII','Halftone','Scanlines','None']){
    await page.getByRole('combobox',{name:'Effect',exact:true}).click();
    await page.getByRole('option',{name:effect,exact:true}).click();
    await settled(effect);
  }
  await page.getByRole('combobox',{name:'Effect',exact:true}).click();await page.getByRole('option',{name:'ASCII',exact:true}).click();await settled('ASCII');
  const workers=await page.evaluate(()=>window.__wallpaperWorkers);
  await page.getByRole('slider',{name:'Visibility',exact:true}).focus();await page.keyboard.press('End');
  await page.getByRole('slider',{name:'Image blur',exact:true}).focus();await page.keyboard.press('ArrowRight');
  await page.getByRole('button',{name:'Horizontal 100%, vertical 0%',exact:true}).click();
  await page.setViewportSize({width:1100,height:800});
  await page.getByRole('textbox',{name:'Message',exact:true}).fill('Streaming and editing do not redraw the background.');
  check(await page.evaluate(()=>window.__wallpaperWorkers)===workers,'CSS controls must not rerun effects');
  await page.reload();await settled('ASCII');
  check(await page.getByRole('slider',{name:'Visibility',exact:true}).getAttribute('aria-valuenow')==='100','visibility persisted');
  check(await page.getByRole('slider',{name:'Image blur',exact:true}).getAttribute('aria-valuenow')==='1','blur persisted');
  check(await page.getByRole('button',{name:'Horizontal 100%, vertical 0%',exact:true}).getAttribute('aria-pressed')==='true','position persisted');
  await upload(true);await page.getByRole('alert').waitFor();await settled('ASCII');
  // Latest-wins changes: only final preferences may survive a reload.
  await page.evaluate(async()=>{const s=window.__wallpaperSmoke;for(let i=0;i<30;i++)s.patch({visibility:i,effect:i===29?'Halftone':'Scanlines'});});
  await settled('Halftone');
  await page.waitForFunction(async()=>{const s=window.__wallpaperSmoke;return s.state().value.visibility===29;});
  // Simulated quota error must preserve the previous committed background.
  await page.evaluate(()=>{window.__put=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==='backgrounds')throw new DOMException('quota','QuotaExceededError');return window.__put.apply(this,args);};});
  await upload();await page.getByRole('alert').waitFor();await settled('Halftone');
  await page.evaluate(()=>{IDBObjectStore.prototype.put=window.__put;delete window.__put;});
  await page.getByRole('button',{name:'Reset adjustments',exact:true}).click();await settled('None');
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
  // Different signed-in account, same origin: artwork must not leak across accounts.
  await page.evaluate(()=>localStorage.setItem('unsloth.browser-account.v1','account:wallpaper-test:other'));
  await page.reload();await page.waitForFunction(()=>document.querySelector('input[type=file]')?.disabled===false);
  check(await page.locator('.chat-wallpaper').count()===0,'account isolation');
  await page.evaluate(()=>localStorage.removeItem('unsloth.browser-account.v1'));
  await page.reload();await settled('Halftone');
  await page.getByRole('button',{name:'Remove background',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('.chat-wallpaper'));
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
