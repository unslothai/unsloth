import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const relative = 'node_modules/@anthropic-ai/sandbox-runtime/dist/sandbox/linux-sandbox-utils.js';
const originalHash = '0ad385c497ab3f0e8b2ad903f90e826e27dd2bdb38cb9714f6a53036ae62c1e8';
const previousHash = '4b9651b7d9249b3f747eb01494af2f31cb4f11216afa93b13250e3d404639a25';
const patchedHash = '6c9566dfb5f3488c9dbb97820466be49b0b534cb657ca8f775ddeadc68b187be';
const original = "args.push('--ro-bind', '/', '/');";
const replacement = "args.push(...(readConfig?.denyOnly?.includes('/') ? ['--tmpfs', '/'] : ['--ro-bind', '/', '/']));";
const sha256 = (data) => createHash('sha256').update(data).digest('hex');

export function applyPatch(root = path.dirname(fileURLToPath(import.meta.url))) {
  const target = path.join(root, relative);
  const bytes = fs.readFileSync(target);
  const hash = sha256(bytes);
  if (hash === patchedHash) return;
  if (hash !== originalHash && hash !== previousHash) throw new Error('Cannot patch an unknown SRT source file');
  const source = hash === previousHash ? bytes.toString('utf8').replace(replacement, original).replaceAll(',fork,reuseaddr,max-children=32', ',fork,reuseaddr') : bytes.toString('utf8');
  if (sha256(source) !== originalHash) throw new Error('Cannot recover the pinned original SRT source');
  if (source.split(original).length !== 2) throw new Error('SRT root-bind patch target is ambiguous');
  let patched = source.replace(original, replacement);
  for (const port of ['3128','1080']) {
    const before=`TCP-LISTEN:${port},fork,reuseaddr`;
    if(patched.split(before).length!==2) throw new Error('SRT socat patch target is ambiguous');
    patched=patched.replace(before,`${before},max-children=32`);
  }
  const nestedBefore = "bwrapArgs.push('--unshare-user', '--bind', '/proc', '/proc');";
  const nestedAfter = "bwrapArgs.push('--unshare-user', '--cap-drop', 'ALL', '--bind', '/proc', '/proc');";
  if (patched.split(nestedBefore).length !== 2) throw new Error('SRT nested capability patch target is ambiguous');
  patched = patched.replace(nestedBefore, nestedAfter);
  if (sha256(patched) !== patchedHash) throw new Error('SRT root-bind patch result mismatch');
  fs.writeFileSync(target, patched, {encoding:'utf8'});
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) applyPatch();
