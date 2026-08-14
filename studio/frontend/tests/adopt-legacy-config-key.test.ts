**.gitattributes**

```makefile
# Python LF normalization
*.py text eol=lf

# TypeScript normalization
*.ts text eol=crlf
```

**studio/frontend/tests/adopt-legacy-config-key.test.ts**

```typescript
import { adoptLegacyConfigKey } from '../../src/config';
import { expect } from 'expect';

describe('adoptLegacyConfigKey', () => {
  it('should adopt legacy snapshot-path config', () => {
    const config = { legacyConfig: { snapshotPath: '/path/to/snapshot' } };
    const adoptedConfig = adoptLegacyConfigKey(config);
    expect(adoptedConfig).toHaveProperty('snapshotPath', '/path/to/snapshot');
  });

  it('should preserve current-id overrides', () => {
    const config = {
      legacyConfig: {
        currentId: 'current-id',
      },
    };
    const adoptedConfig = adoptLegacyConfigKey(config);
    expect(adoptedConfig).toHaveProperty('currentId', 'current-id');
  });

  it('should avoid entry/byte-budget eviction during adoption', () => {
    const config = {
      legacyConfig: {
        entryBudget: 100,
        byteBudget: 100,
      },
    };
    const adoptedConfig = adoptLegacyConfigKey(config);
    expect(adoptedConfig).toHaveProperty('entryBudget', 100);
    expect(adoptedConfig).toHaveProperty('byteBudget', 100);
  });

  it('should handle non-default config key', () => {
    const config = {
      legacyConfig: {
        configKey: 'non-default-key',
      },
    };
    const adoptedConfig = adoptLegacyConfigKey(config);
    expect(adoptedConfig).toHaveProperty('configKey', 'non-default-key');
  });

  it('should handle default config key', () => {
    const config = {
      legacyConfig: {
        configKey: 'default-key',
      },
    };
    const adoptedConfig = adoptLegacyConfigKey(config);
    expect(adoptedConfig).toHaveProperty('configKey', 'default-key');
  });
});