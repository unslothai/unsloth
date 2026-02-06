// src/octto/session/browser.ts
// Cross-platform browser opener

/**
 * Opens the default browser to the specified URL.
 * Detects platform and uses appropriate command.
 */
export async function openBrowser(url: string): Promise<void> {
  const platform = process.platform;

  let command: string[];

  switch (platform) {
    case "darwin":
      command = ["open", url];
      break;
    case "win32":
      command = ["cmd", "/c", "start", url];
      break;
    default:
      // Linux and others
      command = ["xdg-open", url];
      break;
  }

  const proc = Bun.spawn(command, {
    stdout: "ignore",
    stderr: "ignore",
  });

  await proc.exited;
}
