import { countTokens } from '@anthropic-ai/tokenizer';

describe('countTokens', () => {
  test('small text', async () => {
    expect(countTokens('hello world!')).toEqual(3);
  });

  test('text normalising', () => {
    expect(countTokens('™')).toEqual(1);
    expect(countTokens('ϰ')).toEqual(1);
  });

  test('allows special tokens', () => {
    expect(countTokens('<EOT>')).toEqual(1);
  });
});
