// SPDX-License-Identifier: AGPL-3.0-only
import net from 'node:net';
import fs from 'node:fs';

export function assertReadGrantsReleased(result){
  // stillHeld means this holder was removed; another live holder retains access.
  const released = new Set(['revoked','retained','stillHeld','missing','not_found']);
  if(!Array.isArray(result)||result.some(row=>!released.has(row?.status))){
    throw Error('Read grant release was not verified');
  }
}

export async function connectReadLease(transport,request,abort){
  if(process.platform!=='win32'||!Number.isInteger(transport?.port)||transport.port<1||transport.port>65535||!/^[a-f0-9]{64}$/.test(transport.token)||typeof transport.identity!=='string')throw Error('Invalid read lease transport');
  const socket=net.createConnection({host:'127.0.0.1',port:transport.port});
  try{
    await new Promise((resolve,reject)=>{
      socket.once('error',reject);socket.setTimeout(5000,()=>socket.destroy(Error('Read lease timed out')));socket.once('connect',resolve);
    });
    socket.write(JSON.stringify({token:transport.token,identity:transport.identity})+'\n');
    const response=await new Promise((resolve,reject)=>{
      let buffer='';const closed=()=>reject(Error('Read lease closed'));
      socket.once('close',closed);socket.once('error',reject);
      const data=chunk=>{
        buffer+=chunk.toString('utf8');if(buffer.length>262144){socket.destroy();return;}
        if(!buffer.endsWith('\n'))return;
        socket.removeListener('data',data);socket.removeListener('close',closed);
        try{resolve(JSON.parse(buffer));}catch(error){reject(error);}
      };socket.on('data',data);
    });
    const canonical=roots=>roots.map(root=>fs.realpathSync(root).toLowerCase()).sort();
    if(response.identity!==transport.identity||!Array.isArray(response.readRoots)||JSON.stringify(canonical(response.readRoots))!==JSON.stringify(canonical(request.readRoots)))throw Error('Read lease authority mismatch');
    socket.setTimeout(0);socket.on('close',()=>abort.abort());socket.on('error',()=>abort.abort());
    return {socket,readRoots:response.readRoots};
  }catch(error){socket.destroy();throw error;}
}
