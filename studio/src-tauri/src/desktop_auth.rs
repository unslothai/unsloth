fn password_path() -> Result<std::path::PathBuf, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    Ok(home
        .join(".unsloth")
        .join("studio")
        .join("auth")
        .join(".desktop_password"))
}

/// Read the desktop auto-login password from ~/.unsloth/studio/auth/.desktop_password
#[tauri::command]
pub fn get_desktop_password() -> Result<String, String> {
    let path = password_path()?;
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| {
            format!(
                "Failed to read desktop password at {}: {}",
                path.display(),
                e
            )
        })
}

/// Generate a random desktop password in Rust, write it to disk, and return it.
/// The frontend never controls the password content — Rust generates it.
#[tauri::command]
pub fn set_desktop_password() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let auth_dir = home.join(".unsloth").join("studio").join("auth");

    std::fs::create_dir_all(&auth_dir)
        .map_err(|e| format!("Failed to create auth directory: {}", e))?;

    let password: String = (0..64)
        .map(|_| {
            let idx = rand::random_range(0..62u8);
            let c = match idx {
                0..26 => b'a' + idx,
                26..52 => b'A' + (idx - 26),
                52..62 => b'0' + (idx - 52),
                _ => unreachable!(),
            };
            c as char
        })
        .collect();

    let path = auth_dir.join(".desktop_password");
    std::fs::write(&path, password.as_bytes())
        .map_err(|e| format!("Failed to write desktop password: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("Failed to set permissions: {}", e))?;
    }

    Ok(password)
}
