// SPDX-License-Identifier: AGPL-3.0-only
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import readline from 'node:readline';
import {timingSafeEqual} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {verifyInstallation,MAX_REQUEST,errorDiagnostic} from './bridge.mjs';
import {assertReadGrantsReleased} from './windows-read-lease.mjs';

const input=readline.createInterface({input:process.stdin});
let configured=false,stopping=false,server,options,api;
const clients=new Set();
async function stop(){
  if(stopping)return;
  stopping=true;server?.close();
  for(const client of clients)client.destroy();
  try {
    if(options){
      const result=api.revokeWindowsAcl(options);
      assertReadGrantsReleased(result);
    }
  }catch(error){process.exitCode=125;}
  finally{input.close();process.stdin.destroy();}
}
input.on('close',()=>{void stop();});
input.on('line',async line=>{
  if(configured)return;configured=true;
  try{
    if(line.length>MAX_REQUEST)throw Error('Oversized bootstrap');
    const config=JSON.parse(line);
    if(process.platform!=='win32'||!/^[a-f0-9]{64}$/.test(config.token)||typeof config.identity!=='string'||!Array.isArray(config.readRoots)||config.readRoots.length>1024)throw Error('Invalid bootstrap');
    const roots=config.readRoots.map(root=>{
      if(typeof root!=='string'||!path.isAbsolute(root)||/[\x00*?\[\]{}]/.test(root))throw Error('Invalid read root');
      const resolved=fs.realpathSync(root);
      if(resolved===path.parse(resolved).root)throw Error('Root filesystem grant forbidden');
      return resolved;
    });
    const installed=verifyInstallation();
    api=await import(pathToFileURL(path.join(installed,'dist/index.js')).href);
    if(stopping)return;
    const srtWin=api.resolveSrtWin({path:api.VENDORED_SRT_WIN_EXE});
    const user=api.getWindowsSandboxUserStatus({srtWin});
    if(!user.sid)throw Error('Sandbox identity unavailable');
    options={srtWin,sandboxUserSid:user.sid,read:roots,write:[]};
    api.grantWindowsAcl(options);
    server=net.createServer(client=>{
      clients.add(client);client.on('close',()=>clients.delete(client));client.on('error',()=>{});
      client.setTimeout(5000,()=>client.destroy());let buffer='';let admitted=false;
      client.on('data',chunk=>{
        if(admitted){client.destroy();return;}
        buffer+=chunk.toString('utf8');if(buffer.length>1024){client.destroy();return;}
        if(!buffer.endsWith('\n'))return;
        try{
          const hello=JSON.parse(buffer);const expected=Buffer.from(config.token);const supplied=Buffer.from(String(hello.token));
          if(stopping||supplied.length!==expected.length||!timingSafeEqual(supplied,expected)||hello.identity!==config.identity)throw Error('Unauthenticated lease');
          admitted=true;client.setTimeout(0);
          client.write(JSON.stringify({identity:config.identity,readRoots:roots})+'\n');
        }catch(error){client.destroy();}
      });
    });
    await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
    fs.writeSync(1,JSON.stringify({port:server.address().port,sid:user.sid})+'\n');
  }catch(error){
    process.exitCode=125;
    try{fs.writeSync(1,JSON.stringify({error:errorDiagnostic(error)})+'\n');}catch{}
    await stop();
  }
});
