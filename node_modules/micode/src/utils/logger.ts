// src/utils/logger.ts
// Standardized logging with module prefixes
// Used across hooks and tools for consistent log formatting

/**
 * Logger with standardized [module] prefix format.
 *
 * Usage:
 *   log.info("my-hook", "Processing file");
 *   log.error("my-tool", "Failed to read", error);
 *   log.debug("my-module", "Verbose info"); // Only when DEBUG env set
 */
export const log = {
  /**
   * Debug level - only outputs when DEBUG environment variable is set.
   */
  debug(module: string, message: string): void {
    if (process.env.DEBUG) {
      console.log(`[${module}] ${message}`);
    }
  },

  /**
   * Info level - general informational messages.
   */
  info(module: string, message: string): void {
    console.log(`[${module}] ${message}`);
  },

  /**
   * Warning level - non-fatal issues.
   */
  warn(module: string, message: string): void {
    console.warn(`[${module}] ${message}`);
  },

  /**
   * Error level - errors that were caught and handled.
   * @param module - Module name for prefix
   * @param message - Error description
   * @param error - Optional error object for additional context
   */
  error(module: string, message: string, error?: unknown): void {
    if (error !== undefined) {
      console.error(`[${module}] ${message}`, error);
    } else {
      console.error(`[${module}] ${message}`);
    }
  },
};
