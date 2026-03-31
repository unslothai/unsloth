/// Read the desktop auto-login password from ~/.unsloth/studio/auth/.desktop_password
/// Set after first Tauri launch to allow silent re-authentication on subsequent launches.
#[tauri::command]
pub fn get_desktop_password() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let path = home
        .join(".unsloth")
        .join("studio")
        .join("auth")
        .join(".desktop_password");

    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read desktop password at {}: {}", path.display(), e))
}

/// Write the desktop auto-login password to ~/.unsloth/studio/auth/.desktop_password
#[tauri::command]
pub fn set_desktop_password(password: String) -> Result<(), String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let auth_dir = home.join(".unsloth").join("studio").join("auth");

    std::fs::create_dir_all(&auth_dir)
        .map_err(|e| format!("Failed to create auth directory: {}", e))?;

    let path = auth_dir.join(".desktop_password");
    std::fs::write(&path, password.as_bytes())
        .map_err(|e| format!("Failed to write desktop password: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("Failed to set permissions: {}", e))?;
    }

    Ok(())
}
