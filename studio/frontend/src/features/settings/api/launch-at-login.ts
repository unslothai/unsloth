


export async function loadLaunchAtLogin(): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("get_launch_at_login");
}

export async function updateLaunchAtLogin(enabled: boolean): Promise<boolean> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("set_launch_at_login", { enabled });
}
