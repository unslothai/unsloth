# Vendored editor code

The Full rich text composer (`src/features/chat/rich-composer/`) runs on these, copied in rather
than installed so the app takes no new npm dependency. They load only when Settings > Chat >
Rich text is set to Full.

| Folder | Upstream | Version | Upstream commit (npm `gitHead`) | Licence |
| --- | --- | --- | --- | --- |
| `prosemirror-model` | github.com/ProseMirror/prosemirror-model | 1.25.12 | 4211aeeaa3d738454a5d1428a95c1a210e86d787 | MIT |
| `prosemirror-transform` | github.com/ProseMirror/prosemirror-transform | 1.12.4 | 5f56551098afac7bb1d92203408dbfaa22fdf0c0 | MIT |
| `prosemirror-state` | github.com/ProseMirror/prosemirror-state | 1.4.4 | d6fdcd19c4f7f68206b0a8d49649860365672585 | MIT |
| `prosemirror-view` | github.com/ProseMirror/prosemirror-view | 1.42.9 | 2a07ebc03ccfb526b43e60297465129a2cee8ca6 | MIT |
| `prosemirror-keymap` | github.com/ProseMirror/prosemirror-keymap | 1.2.3 | 081f24f65b6a394174a4bdf07b9a55a00361f1b7 | MIT |
| `prosemirror-history` | github.com/ProseMirror/prosemirror-history | 1.5.1 | 44956b6a1929d7bc99aca9a8ec01c790260fa45f | MIT |
| `prosemirror-commands` | github.com/ProseMirror/prosemirror-commands | 1.7.2 | 59fee8d89be2d033b28c2bac68e6736e2989bf9d | MIT |
| `prosemirror-inputrules` | github.com/ProseMirror/prosemirror-inputrules | 1.5.1 | 089aa3852a55d551211af90aae2560c554ccf476 | MIT |
| `orderedmap` | github.com/marijnh/orderedmap | 2.1.1 | 6a30ebbbecd30fed8bda884c7acf4ca9c6a02405 | MIT |
| `rope-sequence` | github.com/marijnh/rope-sequence | 1.3.4 | 11db1ba5aa2abf31ff8040bdf01fabdb9db34c7a | MIT |
| `w3c-keyname` | github.com/marijnh/w3c-keyname | 2.2.8 | 5ea93b04bb62230c7de4165e3a9b5c840f05cd44 | MIT |

Each ProseMirror folder holds the package's published `dist/index.js` and `dist/index.d.ts`
(`prosemirror-view` also `style/prosemirror.css`) and its `LICENSE`, unchanged except for a
one-line header and the imports between the packages, which point at the sibling folders
(`'prosemirror-model'` becomes `'../prosemirror-model/index.js'`) so Vite, `tsc` and
`node --test` all resolve them. `node scripts/vendor-prosemirror.mjs` re-creates them from the
npm tarballs, checking each tarball's integrity hash; to update, change a version and its hash
there and run it.

Biome skips this folder. Not used from upstream: `prosemirror-markdown` (and markdown-it),
`prosemirror-dropcursor`, `prosemirror-gapcursor`, schema-basic and schema-list.
