// SPDX-License-Identifier: AGPL-3.0-only
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {verifyInstallation} from './bridge.mjs';
import {assertReadGrantsReleased} from './windows-read-lease.mjs';
try {
  const input=fs.readFileSync(0,'utf8');
  if(input.length>1024)throw Error('Oversized recovery request');
  const {pid,sid}=JSON.parse(input);
  if(!Number.isInteger(pid)||pid<2||pid===process.pid||typeof sid!=='string'||!/^S-1-[0-9-]+$/.test(sid))throw Error('Invalid holder identity');
  // A recycled PID may now belong to another live owner. Never revoke that owner.
  let alive=true;
  try{process.kill(pid,0);}catch(error){if(error.code==='ESRCH')alive=false;else throw error;}
  if(alive)throw Error('Holder PID is live');
  const root=verifyInstallation();
  const api=await import(pathToFileURL(path.join(root,'dist/index.js')).href);
  const result=api.revokeWindowsAcl({holderPid:pid,sandboxUserSid:sid,srtWin:api.resolveSrtWin({path:api.VENDORED_SRT_WIN_EXE})});
  assertReadGrantsReleased(result);
  fs.writeSync(1,JSON.stringify({released:true})+'\n');
}catch(error){process.exitCode=125;}
