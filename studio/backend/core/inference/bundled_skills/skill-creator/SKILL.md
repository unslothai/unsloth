---
name: skill-creator
description: Create a reusable skill when the user wants Unsloth to learn a repeatable workflow or set of instructions.
allowed-tools: create_skill
---

# Create a skill

Use this skill when the user asks to create, save, or teach Unsloth a reusable workflow.

1. Identify the narrow task the skill should handle and when it should activate.
2. Ask only for missing details that materially change the workflow.
3. Choose a short lowercase name with letters, numbers, and single hyphens.
4. Write a specific description that states when to use the skill.
5. Write concise Markdown instructions with ordered steps, important constraints, and a clear completion check. Do not include secrets, credentials, or machine-specific paths.
6. Call `create_skill` once with the name, description, and complete instructions. It will not overwrite an existing skill.
7. Report the created skill name and suggest invoking it with `@skill-name`.

Prefer one focused skill over a broad collection of unrelated instructions. Do not create scripts or reference files unless the user explicitly asks for them; the creation tool intentionally writes only `SKILL.md`.
