# Anthropic TypeScript Tokenizer

[![NPM version](https://img.shields.io/npm/v/@anthropic-ai/tokenizer.svg)](https://npmjs.org/package/@anthropic-ai/tokenizer)

This package provides a convenient way to check how many tokens a given piece of text will be.

## Installation

```sh
npm install --save @anthropic-ai/tokenizer
# or
yarn add @anthropic-ai/tokenizer
```

## Usage

```js
import { countTokens } from '@anthropic-ai/tokenizer';

function main() {
  const text = 'hello world!';
  const tokens = countTokens(text);
  console.log(`'${text}' is ${tokens} tokens`);
}
main();
```

## Status

This package is in beta. Its internals and interfaces are not stable
and subject to change without a major semver bump;
please reach out if you rely on any undocumented behavior.

We are keen for your feedback; please email us at [support@anthropic.com](mailto:support@anthropic.com)
or open an issue with questions, bugs, or suggestions.

## Requirements

The following runtimes are supported:

- Node.js version 12 or higher.
- Deno v1.28.0 or higher (experimental).
  Use `import { countTokens } from "npm:@anthropic-ai/tokenizer"`.

If you are interested in other runtime environments, please open or upvote an issue on GitHub.
