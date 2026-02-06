// src/mindmodel/formatter.ts
import type { LoadedExample } from "./loader";

export function formatExamplesForInjection(examples: LoadedExample[]): string {
  if (examples.length === 0) return "";

  const blocks = examples.map(
    (ex) => `<example category="${ex.path}" description="${ex.description}">
${ex.content}
</example>`,
  );

  return `<mindmodel-examples>
These are code examples from this project's mindmodel. Follow these patterns when implementing similar functionality.

${blocks.join("\n\n")}
</mindmodel-examples>`;
}
