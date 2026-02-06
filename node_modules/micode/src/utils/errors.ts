// src/utils/errors.ts
// Unified error handling utilities
// Used by tools and hooks for consistent error formatting and logging

/**
 * Safely extract error message from unknown error type.
 * Handles Error instances, strings, and other types.
 */
export function extractErrorMessage(e: unknown): string {
  if (e instanceof Error) {
    return e.message;
  }
  return String(e);
}

/**
 * Format error message for tool responses (LLM-facing).
 * @param message - The error message
 * @param context - Optional context about what operation failed
 */
export function formatToolError(message: string, context?: string): string {
  if (context && context.trim()) {
    return `Error (${context}): ${message}`;
  }
  return `Error: ${message}`;
}

/**
 * Execute a function and log any errors without throwing.
 * Use for non-critical operations that shouldn't fail the main flow.
 * @param module - Module name for log prefix
 * @param fn - Function to execute
 * @returns Result or undefined if error occurred
 */
export function catchAndLog<T>(module: string, fn: () => T): T | undefined {
  try {
    return fn();
  } catch (e) {
    console.error(`[${module}] ${extractErrorMessage(e)}`);
    return undefined;
  }
}

/**
 * Async version of catchAndLog.
 * @param module - Module name for log prefix
 * @param fn - Async function to execute
 * @returns Result or undefined if error occurred
 */
export async function catchAndLogAsync<T>(module: string, fn: () => Promise<T>): Promise<T | undefined> {
  try {
    return await fn();
  } catch (e) {
    console.error(`[${module}] ${extractErrorMessage(e)}`);
    return undefined;
  }
}
