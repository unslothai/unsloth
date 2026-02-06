<h1 align="center">
	<br>
	<br>
	<img width="360" src="logo.png" alt="ulid">
	<br>
	<br>
	<br>
</h1>

# Universally Unique Lexicographically Sortable Identifier

[![Tests](https://github.com/ulid/javascript/actions/workflows/test.yml/badge.svg)](https://github.com/ulid/javascript/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ulid/javascript/branch/master/graph/badge.svg)](https://codecov.io/gh/ulid/javascript)
[![npm](https://img.shields.io/npm/dm/ulid.svg)](https://www.npmjs.com/package/ulid) [![npm](https://img.shields.io/npm/dy/ulid)](https://www.npmjs.com/package/ulid)

ULIDs are unique, sortable identifiers that work much in the same way as UUIDs, though with some improvements:

 * Lexicographically sortable
 * Canonically encoded as a 26 character string, as opposed to the 36 character UUID
 * Uses Crockford's base32 for better efficiency and readability (5 bits per character)
 * Monotonic sort order (correctly detects and handles the same millisecond)

ULIDs also provide:

 * 128-bit compatibility with UUID
 * 1.21e+24 unique IDs per millisecond
 * Case insensitivity
 * No special characters (URL safe)

UUID can be suboptimal for many uses-cases because:

- It isn't the most character efficient way of encoding 128 bits of randomness
- UUID v1/v2 is impractical in many environments, as it requires access to a unique, stable MAC address
- UUID v3/v5 requires a unique seed and produces randomly distributed IDs, which can cause fragmentation in many data structures
- UUID v4 provides no other information than randomness which can cause fragmentation in many data structures

## Installation

Install using NPM:

```shell
npm install ulid --save
```

### Compatibility

ULID supports the following environments:

| Version   | NodeJS    | Browsers      | React-Native  | Web Workers   | Edge Functions    |
|-----------|-----------|---------------|---------------|---------------|-------------------|
| v3        | v18+      | Yes           | Yes           | Yes           | ?                 |
| v2        | v16+      | Yes           | No            | No            | No                |

Additionally, both ESM and CommonJS entrypoints are provided.

## Usage

To quickly generate a ULID, you can simply import the `ulid` function:

```typescript
import { ulid } from "ulid";

ulid(); // "01ARZ3NDEKTSV4RRFFQ69G5FAV"
```

### Seed Time

You can also input a seed time which will consistently give you the same string for the time component. This is useful for migrating to ulid.

```typescript
ulid(1469918176385) // "01ARYZ6S41TSV4RRFFQ69G5FAV"
```

### Monotonic ULIDs

To generate monotonically increasing ULIDs, create a monotonic counter with `monotonicFactory`.

> Note that the same seed time is being passed in for this example to demonstrate its behaviour when generating multiple ULIDs within the same millisecond

```typescript
import { monotonicFactory } from "ulid";

const ulid = monotonicFactory();

// Strict ordering for the same timestamp, by incrementing the least-significant random bit by 1
ulid(150000); // "000XAL6S41ACTAV9WEVGEMMVR8"
ulid(150000); // "000XAL6S41ACTAV9WEVGEMMVR9"
ulid(150000); // "000XAL6S41ACTAV9WEVGEMMVRA"
ulid(150000); // "000XAL6S41ACTAV9WEVGEMMVRB"
ulid(150000); // "000XAL6S41ACTAV9WEVGEMMVRC"

// Even if a lower timestamp is passed (or generated), it will preserve sort order
ulid(100000); // "000XAL6S41ACTAV9WEVGEMMVRD"
```

### Pseudo-Random Number Generators

`ulid` automatically detects a suitable (cryptographically-secure) PRNG. In the browser it will use `crypto.getRandomValues` and on NodeJS it will use `crypto.randomBytes`.

#### Using `Math.random` (insecure)

By default, `ulid` will not use `Math.random` to generate random values. You can bypass this limitation by overriding the PRNG:

```typescript
const ulid = monotonicFactory(() => Math.random());

ulid(); // "01BXAVRG61YJ5YSBRM51702F6M"
```

### Validity

You can verify if a value is a valid ULID by using `isValid`:

```typescript
import { isValid } from "ulid";

isValid("01ARYZ6S41TSV4RRFFQ69G5FAV"); // true
isValid("01ARYZ6S41TSV4RRFFQ69G5FA"); // false
```

### ULID Time

You can encode and decode ULID timestamps by using `encodeTime` and `decodeTime` respectively:

```typescript
import { decodeTime } from "ulid";

decodeTime("01ARYZ6S41TSV4RRFFQ69G5FAV"); // 1469918176385
```

Note that while `decodeTime` works on full ULIDs, `encodeTime` encodes only the _time portion_ of ULIDs:

```typescript
import { encodeTime } from "ulid";

encodeTime(1469918176385); // "01ARYZ6S41"
```

### Tests

Install dependencies using `npm install` first, and then simply run `npm test` to run the test suite.

### CLI

`ulid` can be used on the command line, either via global install:

```shell
npm install -g ulid
ulid
```

Or via `npx`:

```shell
npx ulid
```

You can also generate multiple IDs at the same time:

```shell
ulid --count 15
```

## Specification

You can find the full specification, as well as information regarding implementations in other languages, over at [ulid/spec](https://github.com/ulid/spec).

## Performance

You can test `ulid`'s performance by running `npm run bench`:

```
Simple ulid x 56,782 ops/sec ±2.50% (86 runs sampled)
ulid with timestamp x 58,574 ops/sec ±1.80% (87 runs sampled)
Done!
```
