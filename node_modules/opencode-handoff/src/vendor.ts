/**
 * Code extracted from OpenCode for compatibility.
 *
 * Source: https://github.com/sst/opencode
 * File: packages/opencode/src/tool/read.ts
 *
 * These functions and constants are copied to ensure our synthetic file parts
 * match OpenCode's Read tool output exactly.
 */

import * as path from "node:path"
import * as fs from "node:fs/promises"

/**
 * Constants from OpenCode's ReadTool
 */
export const DEFAULT_READ_LIMIT = 2000
export const MAX_LINE_LENGTH = 2000

/**
 * Binary file extensions (from OpenCode's ReadTool)
 */
const BINARY_EXTENSIONS = new Set([
  ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".class", ".jar", ".war",
  ".7z", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
  ".odp", ".bin", ".dat", ".obj", ".o", ".a", ".lib", ".wasm", ".pyc", ".pyo"
])

/**
 * Check if a file is binary (copied from OpenCode's ReadTool)
 */
export async function isBinaryFile(filepath: string): Promise<boolean> {
  const ext = path.extname(filepath).toLowerCase()

  // Check extension first
  if (BINARY_EXTENSIONS.has(ext)) {
    return true
  }

  try {
    const buffer = await fs.readFile(filepath)
    if (!buffer) return false

    const fileSize = buffer.length
    if (fileSize === 0) return false

    const bufferSize = Math.min(4096, fileSize)
    const bytes = buffer.subarray(0, bufferSize)

    let nonPrintableCount = 0
    for (let i = 0; i < bytes.length; i++) {
      const byte = bytes[i]
      if (byte === undefined) continue
      if (byte === 0) return true
      if (byte < 9 || (byte > 13 && byte < 32)) {
        nonPrintableCount++
      }
    }

    // If >30% non-printable characters, consider it binary
    return nonPrintableCount / bytes.length > 0.3
  } catch {
    return false
  }
}

/**
 * Format file content matching OpenCode's Read tool output format.
 *
 * @param _filepath - Absolute path to the file (unused in output, kept for signature compatibility)
 * @param content - File content as string
 * @returns Formatted output with line numbers in <file> tags
 */
export function formatFileContent(_filepath: string, content: string): string {
  const lines = content.split("\n")
  const limit = DEFAULT_READ_LIMIT
  const offset = 0

  const raw = lines.slice(offset, offset + limit).map((line) => {
    return line.length > MAX_LINE_LENGTH ? line.substring(0, MAX_LINE_LENGTH) + "..." : line
  })

  const formatted = raw.map((line, index) => {
    return `${(index + offset + 1).toString().padStart(5, "0")}| ${line}`
  })

  let output = "<file>\n"
  output += formatted.join("\n")

  const totalLines = lines.length
  const lastReadLine = offset + formatted.length
  const hasMoreLines = totalLines > lastReadLine

  if (hasMoreLines) {
    output += `\n\n(File has more lines. Use 'offset' parameter to read beyond line ${lastReadLine})`
  } else {
    output += `\n\n(End of file - total ${totalLines} lines)`
  }
  output += "\n</file>"

  return output
}
