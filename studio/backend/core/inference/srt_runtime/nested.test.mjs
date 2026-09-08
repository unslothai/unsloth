import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import test from 'node:test';
import {verifyInstallation,quote} from './bridge.mjs';

const enabled=process.platform==='linux' && process.env.UNSLOTH_SRT_NATIVE_TESTS==='1' && process.getuid()===0;
test('root nested wrapper cannot remount an owned read-only runtime grant', {skip:!enabled,timeout:60000}, async()=>{
  const packageRoot=verifyInstallation();
  const {wrapCommandWithSandboxLinux,cleanupBwrapMountPoints}=await import(path.join(packageRoot,'dist/sandbox/linux-sandbox-utils.js'));
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'studio-nested-capabilities-'));
  const work=path.join(root,'work');fs.mkdirSync(work);
  const readonly=path.join(root,'selected-read-only-runtime');fs.mkdirSync(readonly);
  const canary=path.join(readonly,'owned-canary');
  const code=`import ctypes,json,pathlib\np=pathlib.Path(${JSON.stringify(canary)})\nassert p.read_text()=='UNCHANGED'\nlibc=ctypes.CDLL(None,use_errno=True)\nremount=libc.mount(None,${JSON.stringify(readonly)}.encode(),None,4096|32,None)\ntry:p.write_text('MODIFIED');writable=True\nexcept OSError:writable=False\npathlib.Path('positive').write_text('ok')\ncaps=[int(s.split()[1],16) for s in pathlib.Path('/proc/self/status').read_text().splitlines() if s.startswith(('CapEff:','CapPrm:','CapBnd:'))]\nprint(json.dumps({'remount':remount,'writable':writable,'caps':caps}))`;
  try {
    for(const nested of [false,true]) {
      fs.writeFileSync(canary,'UNCHANGED');
      const wrapped=await wrapCommandWithSandboxLinux({command:'/usr/bin/python3 -I -S -c '+quote(code),needsNetworkRestriction:true,readConfig:{denyOnly:['/','/sys'],allowWithinDeny:['/usr','/bin','/lib','/lib64',readonly]},writeConfig:{allowOnly:[work],denyWithinAllow:[]},binShell:'/bin/bash',bwrapPath:'/usr/bin/bwrap',allowAllUnixSockets:true,enableWeakerNestedSandbox:nested});
      const result=spawnSync('/bin/bash',['-c',wrapped],{cwd:work,encoding:'utf8',timeout:15000});
      assert.equal(result.status,0,result.stderr);
      const observed=JSON.parse(result.stdout);
      assert.equal(observed.remount,-1);
      assert.equal(observed.writable,false);
      assert.deepEqual(observed.caps,[0,0,0]);
      assert.equal(fs.readFileSync(canary,'utf8'),'UNCHANGED');
      assert.equal(fs.readFileSync(path.join(work,'positive'),'utf8'),'ok');
      cleanupBwrapMountPoints();
    }
  } finally {cleanupBwrapMountPoints();fs.rmSync(root,{recursive:true,force:true});}
});
