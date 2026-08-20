


/** Local models (LM Studio, Ollama, custom folders) are not fine-tuned;
 * they live in the Hub picker's On Device section. */
export function isFineTunedSource(source?: string): boolean {
  return source !== "local";
}
