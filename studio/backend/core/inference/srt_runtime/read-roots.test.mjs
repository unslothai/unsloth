import assert from 'node:assert/strict';
import path from 'node:path';
import { test } from 'node:test';
import { validateRequest } from './bridge.mjs';

const roots = (count) => Array.from({length:count}, (_,i) => path.resolve(`/selected/lib/libfixture${i}.so`));
const request = (readRoots) => ({v:1,operation:'probe',executable:path.resolve('/selected/bin/python'),argv:[],cwd:path.resolve('/work'),env:{},readRoots,writeRoots:[path.resolve('/work')],timeoutMs:1000});

test('accepts the Colab-sized explicit library list without replacing or dropping paths', () => {
  const selected = roots(143);
  assert.deepEqual(validateRequest(request(selected)).readRoots, selected);
});

test('read-root count has a bounded limit and a count-specific diagnostic', () => {
  assert.equal(validateRequest(request(roots(1024))).readRoots.length, 1024);
  assert.throws(() => validateRequest(request(roots(1025))), /readRoots exceeds 1024 entries/);
});

test('large read lists still reject invalid paths including the final entry', () => {
  for (const invalid of ['/', 'relative/lib.so', '/selected/*', '/selected/x\0y', null]) {
    assert.throws(() => validateRequest(request([...roots(142), invalid])), /explicit absolute paths/);
  }
});

test('write and deny lists retain their smaller authority limits', () => {
  for (const key of ['writeRoots','denyReadRoots','denyWriteRoots']) {
    assert.doesNotThrow(() => validateRequest({...request(roots(143)), [key]:roots(128)}));
    assert.throws(() => validateRequest({...request(roots(143)), [key]:roots(129)}), new RegExp(`${key} exceeds 128 entries`));
  }
});
