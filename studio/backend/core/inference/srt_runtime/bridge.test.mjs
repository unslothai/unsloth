import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { MAX_REQUEST, payloadCommand, quote, validateRequest, verifyInstallation } from './bridge.mjs';
import { applyPatch } from './apply_patch.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const request = () => ({v:1,operation:'run',executable:path.resolve('/usr/bin/python3'),argv:['-c','print(1)'],cwd:path.resolve('/tmp/work'),env:{PATH:'/usr/bin:/bin'},readRoots:[path.resolve('/usr')],writeRoots:[path.resolve('/tmp/work')],timeoutMs:1000});

test('accepts explicit argv and rejects authority fields, glob grants and root grants', () => {
  assert.equal(validateRequest(request()).v, 1);
  assert.equal(validateRequest({...request(),timeoutMs:null}).timeoutMs,null);
  assert.equal(validateRequest({...request(),privateUnixSockets:true}).privateUnixSockets,true);
  assert.throws(()=>validateRequest({...request(),privateUnixSockets:'true'}));
  for (const update of [{allowedDomains:['example.com']},{allowAllUnixSockets:true},{readRoots:['/']},{writeRoots:['/tmp/*']},{denyReadRoots:['/tmp/*']},{denyReadRoots:['/']},{argv:['x\0y']},{timeoutMs:0},{v:2}]) {
    assert.throws(() => validateRequest({...request(),...update}));
  }
});

test('rejects shell and native loader environment injection', () => {
  for (const key of ['LD_PRELOAD','LD_LIBRARY_PATH','BASH_ENV','ENV','NODE_OPTIONS','PYTHONSTARTUP']) {
    assert.throws(() => validateRequest({...request(),env:{[key]:'attacker'}}));
  }
});

test('proxy socket paths cannot inject the upstream socat shell command',()=>{
  for(const p of ['/tmp/socket;touch-payload','/tmp/$(payload)','/tmp/socket\nexec','/tmp/socket space','/tmp/../socket']) {
    assert.throws(()=>validateRequest({...request(),network:{httpSocketPath:p,socksSocketPath:'/tmp/socks'}}));
  }
  assert.throws(()=>validateRequest({...request(),network:{httpSocketPath:'/tmp/http',socksSocketPath:'/tmp/socks',proxyAuthToken:'secret'}}));
});

test('POSIX quoting survives command substitutions and metacharacters', {skip:process.platform==='win32'}, () => {
  const values = ["a'b", '$(printf INJECTED)', '`printf INJECTED`', 'x\ny', 'a; echo wrong', ''];
  const run = spawnSync('/bin/bash',['-c',`printf '%s\\0' ${values.map(quote).join(' ')}`]);
  assert.equal(run.status,0);
  assert.deepEqual(run.stdout.toString().split('\0').slice(0,-1),values);
});

test('published files match the pinned integrity ledger', () => {
  assert.match(verifyInstallation(), /sandbox-runtime$/);
});

test('network payload uses numeric proxy endpoints without hostname resolution', {skip:process.platform==='win32'}, () => {
  const script="printf '%s\\n' \"$HTTPS_PROXY\" \"$http_proxy\" \"$ALL_PROXY\" \"$NO_PROXY\" \"$1\"";
  const command=payloadCommand({network:{},executable:'/bin/bash',argv:['-c',script,'payload','$(touch SHOULD_NOT_EXIST)']});
  const run=spawnSync('/bin/bash',['-c',command],{env:{HTTPS_PROXY:'http://localhost:3128',NO_PROXY:'pypi.org'}});
  assert.equal(run.status,0);
  assert.deepEqual(run.stdout.toString().trim().split('\n'),['http://127.0.0.1:3128','http://127.0.0.1:3128','http://127.0.0.1:3128','localhost,127.0.0.1,::1','$(touch SHOULD_NOT_EXIST)']);
});

test('installation patch is idempotent and refuses an unknown upstream source',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-patch-'));
  const relative='node_modules/@anthropic-ai/sandbox-runtime/dist/sandbox/linux-sandbox-utils.js';
  const target=path.join(dir,relative);fs.mkdirSync(path.dirname(target),{recursive:true});
  try {
    const expected=fs.readFileSync(path.join(here,relative),'utf8');
    const original=expected.replace("args.push(...(readConfig?.denyOnly?.includes('/') ? ['--tmpfs', '/'] : ['--ro-bind', '/', '/']));","args.push('--ro-bind', '/', '/');").replaceAll(',fork,reuseaddr,max-children=32',',fork,reuseaddr');
    fs.writeFileSync(target,original);applyPatch(dir);
    assert.equal(fs.readFileSync(target,'utf8'),expected);
    applyPatch(dir);assert.equal(fs.readFileSync(target,'utf8'),expected);
    fs.appendFileSync(target,'\nUNKNOWN_UPSTREAM_CHANGE');
    assert.throws(()=>applyPatch(dir),/unknown SRT source/);
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});

test('malformed requests fail on the dedicated channel before any spawn', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-protocol-'));
  const control = path.join(dir,'control');
  try {
    for (const input of ['not-json', JSON.stringify({...request(),filesystem:{disabled:true}}), ' '.repeat(MAX_REQUEST+1)]) {
      const fd = fs.openSync(control,'w');
      let run;
      try {run = spawnSync(process.execPath,[path.join(here,'bridge.mjs'),'3'],{input,stdio:['pipe','pipe','pipe',fd],timeout:10000});}
      finally {fs.closeSync(fd);}
      assert.equal(run.status,125);
      assert.equal(run.stdout.length,0);
      const records=fs.readFileSync(control,'utf8').trim().split('\n').map(JSON.parse);
      assert.equal(records.length,1);
      assert.equal(records[0].event,'error');
    }
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});

test('an altered integrity ledger cannot waive package verification', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-integrity-'));
  try {
    fs.writeFileSync(path.join(dir,'integrity.json'),JSON.stringify({version:'0.0.75',files:{}}));
    assert.throws(() => verifyInstallation(dir));
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});

test('missing or corrupted installed files fail before the payload', {skip:process.platform!=='linux'}, () => {
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-broken-'));
  const work=path.join(dir,'work');fs.mkdirSync(work);
  const copy=path.join(dir,'runtime');fs.mkdirSync(copy);
  const control=path.join(dir,'control');
  fs.copyFileSync(path.join(here,'bridge.mjs'),path.join(copy,'bridge.mjs'));
  fs.copyFileSync(path.join(here,'integrity.json'),path.join(copy,'integrity.json'));
  const first=Object.keys(JSON.parse(fs.readFileSync(path.join(copy,'integrity.json'),'utf8')).files)[0];
  try {
    for(const corrupt of [false,true]) {
      if(corrupt){const target=path.join(copy,'node_modules',first);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,'CORRUPTED_PACKAGE_FILE');}
      const fd=fs.openSync(control,'w');
      let run;
      try {run=spawnSync(process.execPath,[path.join(copy,'bridge.mjs'),'3'],{input:JSON.stringify({...request(),cwd:work,writeRoots:[work],argv:['-c',"open('MUST_NOT_RUN','w').write('bad')"]}),stdio:['pipe','pipe','pipe',fd],timeout:10000});}
      finally {fs.closeSync(fd);}
      assert.equal(run.status,125);
      assert.equal(fs.existsSync(path.join(work,'MUST_NOT_RUN')),false);
      const records=fs.readFileSync(control,'utf8').trim().split('\n').map(JSON.parse);
      assert.equal(records.length,1);assert.equal(records[0].event,'error');
      if(corrupt)assert.match(records[0].message,/integrity mismatch/);
    }
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});

test('host temp grants and symlink aliases fail before payload execution', {skip:process.platform!=='linux'}, () => {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'unsloth-srt-temp-grant-'));
  const work=path.join(root,'work');fs.mkdirSync(work);
  const alias=path.join(root,'tmp-alias');fs.symlinkSync('/tmp',alias);
  const marker=path.join(work,'MUST_NOT_RUN');
  const control=path.join(root,'control');
  try {
    for(const target of ['/tmp',alias]) {
      for(const update of [{cwd:target},{readRoots:[target]},{writeRoots:[target]}]) {
        const fd=fs.openSync(control,'w');let run;
        try {
          const value={...request(),cwd:work,writeRoots:[work],argv:['-c',`open(${JSON.stringify(marker)},'w').write('bad')`],...update};
          run=spawnSync(process.execPath,[path.join(here,'bridge.mjs'),'3'],{input:JSON.stringify(value),stdio:['pipe','pipe','pipe',fd],timeout:10000});
        } finally {fs.closeSync(fd);}
        assert.equal(run.status,125);
        assert.equal(fs.existsSync(marker),false);
        const records=fs.readFileSync(control,'utf8').trim().split('\n').map(JSON.parse);
        assert.equal(records.length,1);assert.equal(records[0].event,'error');
        assert.match(records[0].message,/Host \/tmp cannot replace/);
      }
    }
  } finally {fs.rmSync(root,{recursive:true,force:true});}
});
