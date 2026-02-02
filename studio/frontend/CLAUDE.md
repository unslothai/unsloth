# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM fine-tuning platform UI (Unsloth-branded). React/TypeScript frontend with a skeletal Python backend. The frontend is the active development focus.

## Commands

All commands run from `frontend/`:

```bash
bun install          # install dependencies
bun run dev          # start Vite dev server
bun run build        # typecheck + production build
bun run typecheck    # TypeScript type checking only
bun run lint         # ESLint
bun run biome:check  # Biome linter + formatter check
bun run biome:fix    # Biome auto-fix
```

Package manager is **Bun** (not npm/yarn).

## Architecture

### Frontend (`frontend/src/`)


**Feature-based module architecture** with enforced boundaries:

- `features/` — self-contained feature modules (chat, onboarding, studio)
- `components/ui/` — shadcn/ui primitives (linting/formatting disabled for these)
- `components/assistant-ui/` — AI chat thread components
- `components/layout/` — layout shells (dashboard)
- `stores/` — Zustand stores (training wizard state)
- `config/` — constants (model lists, hyperparameters, env)
- `types/` — shared TypeScript types
- `app/` — router and root layout (TanStack React Router)

### Import Rules (ESLint-enforced)

Cross-feature imports are **prohibited**. Import from feature barrel (`@/features/[name]`), never from internal paths (`@/features/chat/some-component`).

### Key Technology Choices

| Concern | Choice |
|---------|--------|
| Routing | TanStack React Router |
| State | Zustand |
| Styling | Tailwind CSS + shadcn/ui (radix-maia style, HugeIcons) |
| Animation | Framer Motion |
| Chat UI | @assistant-ui/react with streaming |
| Local DB | Dexie (IndexedDB) for chat threads/messages |
| Charts | Recharts |

### Backend (`backend/`)

Placeholder Python structure. Frontend expects an inference server at the URL in `frontend/.env` (`VITE_INFERENCE_URL`) serving POST `/api/chat/generate` with streaming responses.

## Code Style

- Biome handles formatting (2-space indent) and import organization
- `src/components/ui/**` is excluded from Biome linting/formatting (generated shadcn code)
- Path alias: `@` maps to `frontend/src/`
- Prefer KISS and DRY
