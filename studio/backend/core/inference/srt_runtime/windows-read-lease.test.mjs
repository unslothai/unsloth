import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import net from 'node:net';
import {connectReadLease,assertReadGrantsReleased} from './windows-read-lease.mjs';

test('shared read release succeeds without hiding cleanup failures',()=>{
  assert.doesNotThrow(()=>assertReadGrantsReleased([{status:'stillHeld'},{status:'revoked'}]));
  for(const result of [undefined,null,{},[null],[{status:'failed'}],[{status:'mismatch'}],[{status:'unknown'}]]){
    assert.throws(()=>assertReadGrantsReleased(result));
  }
});

test('read lease accepts exact authority and aborts when the owner connection closes',{skip:process.platform!=='win32'},async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'read-lease-wire-'));
  let peer;
  const server=net.createServer(socket=>{
    peer=socket;socket.once('data',()=>socket.write(JSON.stringify({identity:'runtime',readRoots:[root]})+'\n'));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  try{
    const abort=new AbortController();
    const lease=await connectReadLease({port:server.address().port,token:'a'.repeat(64),identity:'runtime'},{readRoots:[root]},abort);
    const closed=new Promise(resolve=>abort.signal.addEventListener('abort',resolve,{once:true}));
    peer.destroy();await closed;assert.equal(abort.signal.aborted,true);lease.socket.destroy();
  }finally{server.close();peer?.destroy();fs.rmdirSync(root);}
});

test('mismatched identity or roots never produce a usable lease',{skip:process.platform!=='win32'},async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'read-lease-deny-'));
  try{
    for(const reply of [{identity:'other',readRoots:[root]},{identity:'runtime',readRoots:[]},{identity:'runtime',readRoots:'invalid'}]){
      const server=net.createServer(socket=>socket.once('data',()=>socket.end(JSON.stringify(reply)+'\n')));
      await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
      try{await assert.rejects(connectReadLease({port:server.address().port,token:'a'.repeat(64),identity:'runtime'},{readRoots:[root]},new AbortController()));}
      finally{server.close();}
    }
  }finally{fs.rmdirSync(root);}
});
