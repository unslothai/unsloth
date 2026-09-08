import assert from 'node:assert/strict';
import test from 'node:test';
import {validateRequest, errorDiagnostic, verifyInstallation, linuxDependencies} from './bridge.mjs';
import fs from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';

test('policy failures are structured without including input', () => {
  try {validateRequest({secret:'private-token'}); assert.fail('accepted');}
  catch (error) {
    assert.deepEqual(errorDiagnostic(error), {code:'policy_invalid',stage:'policy'});
    assert.doesNotMatch(JSON.stringify(errorDiagnostic(error)), /private-token/);
  }
});
test('missing installation has an installation reason', () => {
  assert.throws(() => verifyInstallation(path.join(tmpdir(),'unsloth-no-installed-runtime-fixture')), error => {
    assert.deepEqual(errorDiagnostic(error), {code:'runtime_missing',stage:'installation'});
    return true;
  });
});
test('unknown errors never turn into container eligibility or expose messages', () => {
  assert.deepEqual(errorDiagnostic(new Error('bwrap: mount proc: secret')), {code:'probe_failed',stage:'probe'});
});
test('unqualified nested settings cannot enter the execution protocol', () => {
  for(const extra of [{enableWeakerNestedSandbox:true},{isolationVariant:'nested'}]) {
    assert.throws(()=>validateRequest({v:1,operation:'run',...extra}),error=>errorDiagnostic(error).code==='policy_invalid');
  }
});

test('Linux checks both dependencies and resolves an absolute ripgrep fallback', t => {
  let available = new Set();
  t.mock.method(fs, 'accessSync', filename => { if (!available.has(filename)) throw new Error('missing fixture'); });
  t.mock.method(fs, 'statSync', () => ({isFile:()=>true}));
  assert.throws(()=>linuxDependencies(), error=>errorDiagnostic(error).dependency==='bubblewrap');
  available.add('/usr/bin/bwrap');
  assert.throws(()=>linuxDependencies(), error=>errorDiagnostic(error).dependency==='ripgrep');
  available.add('/usr/local/bin/rg');
  assert.deepEqual(linuxDependencies(), {bubblewrap:'/usr/bin/bwrap',ripgrep:'/usr/local/bin/rg'});
  available.add('/usr/bin/rg');
  assert.equal(linuxDependencies().ripgrep, '/usr/bin/rg');
});
