import test from 'node:test';
import assert from 'node:assert/strict';
import { validateRequest, executeSupported } from './bridge.mjs';

test('Linux and macOS never select the Studio SRT bridge', async () => {
  for (const platform of ['linux', 'darwin']) {
    await assert.rejects(executeSupported({}, () => {}, { platform }), /Unsupported SRT platform/);
  }
});

test('removed nested selection and Linux proxy fields are rejected', () => {
  for (const key of ['isolationVariant', 'network']) {
    assert.throws(() => validateRequest({ v: 1, operation: 'run', [key]: 'nested' }), /Unknown SRT request field/);
  }
});
