#!/usr/bin/env yarn tsn -T

import { countTokens } from '@anthropic-ai/tokenizer';

function main() {
  const text = 'hello world!';
  const tokens = countTokens(text);
  console.log(`'${text}' is ${tokens} tokens`);
}

main();
