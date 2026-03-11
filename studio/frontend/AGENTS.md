# Repository Guidelines

## Project Structure & Module Organization
- `src/` is app code; entry is `src/main.tsx`, global styles in `src/index.css`.
- `src/app/` holds app shell and routing; `src/features/` is feature slices w/ public `index.ts` exports.
- Shared UI lives in `src/components/` (shadcn in `src/components/ui/`).
- Shared logic in `src/hooks/`, `src/stores/`, `src/utils/`, `src/lib/`, and types in `src/types/`.
- Static assets: `src/assets/` and `public/`.
- `test/` is a Python harness for payload validation and preview; not a JS test suite.

## Build, Test, and Development Commands
- `bun run dev`: start Vite dev server.
- `bun run build`: typecheck + build to `dist/`.
- `bun run preview`: serve the production build locally.
- `bun run lint`: ESLint checks for TS/React.
- `bun run typecheck`: `tsc` no-emit verification.
- `bun run biome:check` / `bun run biome:fix`: format + lint w/ Biome.
- Optional harness: `python test/scripts/validate_payload.py test/data/ui_payload.json`.

## Coding Style & Naming Conventions
- TypeScript + React, 2-space indent (Biome).
- Prefer explicit, compact code; avoid heavy abstraction.
- Use path alias `@/` for app imports.
- Feature boundaries enforced: import from `@/features/<name>` only, not deep paths.
- Components in `PascalCase`, hooks in `useCamelCase`, files in `kebab-case` or `camelCase` per local convention.

## Testing Guidelines
- No frontend test runner configured yet; add one if needed.
- `test/` is for API payload validation and preview flows; add samples as `test/data/ui_payload_*.json`.

## Commit & Pull Request Guidelines
- Commit history shows short, imperative messages; optional prefix like `refactor:`; keep it terse.
- PRs should include: clear summary, linked issue (if any), and UI screenshots/gifs for visual changes.
- Call out new deps, config, or required env changes in the PR body.

## Agent Notes
- Keep changes minimal, focused, and easy to review.
