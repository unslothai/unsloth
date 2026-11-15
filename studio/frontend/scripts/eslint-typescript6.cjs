// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Imported first by eslint.config.js so CLI and editors both get it; typescript-eslint needs the
// TypeScript JS API, which TypeScript 7 lacks, so require('typescript') gets the TypeScript 6 copy
// from @typescript/typescript6. Drop once typescript-eslint supports TS 7.

const Module = require("node:module");
const path = require("node:path");

const ts6Require = Module.createRequire(
  require.resolve("@typescript/typescript6", {
    paths: [path.join(__dirname, "..")],
  }),
);
const ts6Root = path.dirname(
  ts6Require.resolve("@typescript/old/package.json"),
);
const resolveFilename = Module._resolveFilename;

Module._resolveFilename = function (request, parent, isMain, options) {
  if (request === "typescript" || request.startsWith("typescript/")) {
    return resolveFilename.call(
      this,
      ts6Root + request.slice("typescript".length),
      parent,
      isMain,
      options,
    );
  }
  return resolveFilename.call(this, request, parent, isMain, options);
};
