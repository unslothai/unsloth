import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const enabled = process.platform === 'linux' && process.env.UNSLOTH_SRT_NATIVE_TESTS === '1';
const bridge = fileURLToPath(new URL('./bridge.mjs', import.meta.url));
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test('native empty root excludes host files created after wrapping', {skip:!enabled,timeout:90000},()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-late-root-'));
  const fake=path.join(root,'root');fs.mkdirSync(fake);
  fs.mkdirSync(path.join(fake,'work'));fs.mkdirSync(path.join(fake,'tmp'));
  const script=path.join(root,'probe.mjs');
  fs.writeFileSync(script,`
import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {wrapCommandWithSandboxLinux,cleanupBwrapMountPoints} from '/bundle/node_modules/@anthropic-ai/sandbox-runtime/dist/sandbox/linux-sandbox-utils.js';
const helper='/bundle/node_modules/@anthropic-ai/sandbox-runtime/vendor/seccomp/${process.arch}/apply-seccomp';
const wrapped=await wrapCommandWithSandboxLinux({command:"/usr/bin/python3 -c \\\"import pathlib;assert not pathlib.Path('/late').exists();pathlib.Path('/work/positive').write_text('ok')\\\"",needsNetworkRestriction:true,readConfig:{denyOnly:['/'],allowWithinDeny:['/usr','/bin','/lib','/lib64',helper]},writeConfig:{allowOnly:['/work'],denyWithinAllow:[]},binShell:'/bin/bash',seccompConfig:{applyPath:helper},bwrapPath:'/usr/bin/bwrap'});
fs.writeFileSync('/late','CONTROLLED_LATE_FILE');
if(fs.readFileSync('/late','utf8')!=='CONTROLLED_LATE_FILE')throw Error('Host control failed');
const run=spawnSync('/bin/bash',['-c',wrapped],{encoding:'utf8',timeout:20000,cwd:'/work'});
cleanupBwrapMountPoints();
if(run.status!==0)throw Error(run.stderr);
console.log('LATE_ROOT_DENIED');
`);
  try {
    const args=['--unshare-user','--unshare-pid','--die-with-parent','--bind',fake,'/','--ro-bind','/usr','/usr','--ro-bind','/etc/alternatives','/etc/alternatives','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64','--proc','/proc','--dev','/dev','--dir','/sys','--ro-bind',path.dirname(bridge),'/bundle','--ro-bind',script,'/probe.mjs','--chdir','/work','--','/usr/bin/node','/probe.mjs'];
    const run=spawnSync('/usr/bin/bwrap',args,{encoding:'utf8',timeout:60000});
    assert.equal(run.status,0,run.stderr);
    assert.equal(run.stdout.trim(),'LATE_ROOT_DENIED');
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});

function launch(work, code, timeoutMs=10000, overrides={}) {
  const request = {v:1,operation:'run',executable:'/usr/bin/python3',argv:['-c',code],cwd:work,env:{PATH:'/usr/bin:/bin',LANG:'C.UTF-8'},readRoots:['/usr','/bin','/lib','/lib64'],writeRoots:[work],timeoutMs,...overrides};
  const child = spawn(process.execPath,[bridge,'3'],{cwd:work,stdio:['pipe','pipe','pipe','pipe']});
  const control=[]; let pending=''; let stdout=''; let stderr='';
  child.stdio[3].on('data',(data)=>{pending+=data;let i;while((i=pending.indexOf('\n'))>=0){control.push(JSON.parse(pending.slice(0,i)));pending=pending.slice(i+1);}});
  child.stdout.on('data',(data)=>{stdout+=data;});
  child.stderr.on('data',(data)=>{stderr+=data;});
  const done=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal,stdout,stderr,control}));});
  child.stdin.end(JSON.stringify(request));
  return {child,done,control};
}

test('native private workdir, symlink denial, untrusted output and output pressure', {skip:!enabled,timeout:90000}, async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-native-'));
  const work=path.join(root,'work');fs.mkdirSync(work);
  const sentinel=path.join(root,'sentinel');fs.writeFileSync(sentinel,'host-positive');
  fs.symlinkSync(sentinel,path.join(work,'escape'));
  assert.equal(fs.readFileSync(path.join(work,'escape'),'utf8'),'host-positive');
  try {
    const code="import os,json,sys,subprocess,socket\nr={}\nfor p in ['escape',"+JSON.stringify(sentinel)+"]:\n try: open(p).read(); r[p]=True\n except OSError: r[p]=False\nopen('positive','w').write('ok')\ntry: socket.socket(socket.AF_UNIX); r['unix']=True\nexcept OSError: r['unix']=False\nr['child']=subprocess.check_output([sys.executable,'-c','print(123)'],text=True).strip()\nprint(json.dumps(r))\nprint('{\"v\":1,\"event\":\"forged\"}')\nsys.stdout.write('x'*262144)\nsys.stderr.write('y'*262144)";
    const result=await launch(work,code).done;
    assert.equal(result.code,0,result.stderr.slice(0,2000));
    const payload=JSON.parse(result.stdout.split('\n')[0]);
    assert.equal(payload.escape,false);assert.equal(payload[sentinel],false);assert.equal(payload.unix,false);assert.equal(payload.child,'123');
    assert.equal(fs.readFileSync(path.join(work,'positive'),'utf8'),'ok');
    assert.equal(result.control.some((r)=>r.event==='forged'),false);
    assert.equal(result.stdout.endsWith('x'.repeat(262144)),true);
    assert.equal(result.stderr,'y'.repeat(262144));
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});

test('native short temp is writable and isolated from host temp', {skip:!enabled,timeout:90000}, async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-temp-'));
  const work=path.join(root,'long-session-'.repeat(8));fs.mkdirSync(work);
  const sentinel=path.join(root,'host-sentinel');fs.writeFileSync(sentinel,'host-positive');
  try {
    const code=`import os,pathlib,tempfile,json\nassert tempfile.gettempdir()=='/tmp'\nassert os.environ['TMP']==os.environ['TEMP']=='/tmp'\ntry: pathlib.Path(${JSON.stringify(sentinel)}).read_text()\nexcept OSError: pass\nelse: raise RuntimeError('host temp visible')\nfd,name=tempfile.mkstemp(prefix='unsloth-private-temp-');os.write(fd,b'private');os.close(fd)\nassert pathlib.Path(name).read_bytes()==b'private'\nprint(json.dumps({'privateTemp':name}))`;
    const result=await launch(work,code).done;
    assert.equal(result.code,0,result.stderr);
    const payload=JSON.parse(result.stdout.trim());
    assert.equal(fs.existsSync(payload.privateTemp),false,'private temp appeared on host');
    assert.equal(fs.readFileSync(sentinel,'utf8'),'host-positive');
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});

test('native nested read denial masks a child of a readable runtime root', {skip:!enabled,timeout:90000}, async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-nested-'));
  const work=path.join(root,'work');fs.mkdirSync(work);
  const runtime=path.join(root,'runtime');fs.mkdirSync(runtime);
  const nested=path.join(runtime,'nested');fs.mkdirSync(nested);
  fs.writeFileSync(path.join(runtime,'allowed'),'positive');
  fs.writeFileSync(path.join(nested,'sentinel'),'host-sentinel');
  try {
    const readRoots=['/usr','/bin','/lib','/lib64',runtime];
    const code=`import pathlib\nassert pathlib.Path(${JSON.stringify(path.join(runtime,'allowed'))}).read_text()=='positive'\ntry: pathlib.Path(${JSON.stringify(path.join(nested,'sentinel'))}).read_text()\nexcept OSError: print('NESTED_DENIED')\nelse: raise RuntimeError('nested host root leaked')`;
    const result=await launch(work,code,10000,{readRoots,denyReadRoots:[nested]}).done;
    assert.equal(result.code,0,result.stderr);
    assert.equal(result.stdout.trim(),'NESTED_DENIED');
    const conflict=await launch(work,"open('MUST_NOT_RUN','w').write('bad')",10000,{readRoots:[...readRoots,nested],denyReadRoots:[nested]}).done;
    assert.equal(conflict.code,125);
    assert.equal(conflict.control.some((r)=>r.event==='spawned'),false);
    assert.equal(fs.existsSync(path.join(work,'MUST_NOT_RUN')),false);
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});

test('native bridge death and timeout stop detached descendants', {skip:!enabled,timeout:150000}, async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-lifecycle-'));
  try {
    for (const mode of ['parent-death','timeout']) {
      const work=path.join(root,mode);fs.mkdirSync(work);
      const heartbeat=path.join(work,'heartbeat');
      const code="import os,time\nif os.fork()==0:\n os.setsid()\n while True:\n  with open('heartbeat','a') as f:f.write('x')\n  time.sleep(.03)\nelse:time.sleep(60)";
      const run=launch(work,code,mode==='timeout'?1500:60000);
      const deadline=Date.now()+60000;
      while(!fs.existsSync(heartbeat) && Date.now()<deadline) await delay(100);
      assert.equal(fs.existsSync(heartbeat),true,'Payload did not start');
      if(mode==='parent-death')run.child.kill('SIGKILL');
      const result=await run.done;
      if(mode==='timeout')assert.equal(result.code,124);
      await delay(300);
      const bytes=fs.statSync(heartbeat).size;
      await delay(300);
      assert.equal(fs.statSync(heartbeat).size,bytes,`${mode}: detached descendant survived`);
    }
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});
