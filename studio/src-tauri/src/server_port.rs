use log::warn;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::Manager;

const PREFERENCE_FILE: &str = "server-port-v1";
pub const AUTOMATIC_PORT: u16 = 8888;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchPolicy {
    pub port: u16,
    pub exact: bool,
}

fn preference_path(config_dir: &Path) -> PathBuf {
    config_dir.join(PREFERENCE_FILE)
}

fn read_preference(config_dir: &Path) -> Option<u16> {
    let value = fs::read_to_string(preference_path(config_dir)).ok()?;
    match value.trim() {
        "automatic" => None,
        value => value.parse::<u16>().ok().filter(|port| *port > 0),
    }
}

fn write_preference(config_dir: &Path, port: Option<u16>) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let value = port.map_or_else(|| "automatic".to_string(), |port| port.to_string());
    let path = preference_path(config_dir);
    fs::write(&path, format!("{value}\n")).map_err(|error| {
        format!(
            "Failed to save server port preference {}: {error}",
            path.display()
        )
    })
}

fn validate_port(port: Option<u32>) -> Result<Option<u16>, String> {
    match port {
        None => Ok(None),
        Some(1..=65535) => Ok(port.map(|value| value as u16)),
        Some(_) => Err("Server port must be between 1 and 65535.".to_string()),
    }
}

fn policy_for(port: Option<u16>) -> LaunchPolicy {
    match port {
        Some(port) => LaunchPolicy { port, exact: true },
        None => LaunchPolicy {
            port: AUTOMATIC_PORT,
            exact: false,
        },
    }
}

fn app_config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

pub fn launch_policy(app: &tauri::AppHandle) -> LaunchPolicy {
    match app_config_dir(app) {
        Ok(dir) => policy_for(read_preference(&dir)),
        Err(error) => {
            warn!("{error}; using automatic server port selection");
            policy_for(None)
        }
    }
}

#[tauri::command]
pub fn get_server_port(app: tauri::AppHandle) -> Option<u16> {
    app_config_dir(&app)
        .ok()
        .and_then(|dir| read_preference(&dir))
}

#[tauri::command]
pub fn set_server_port(app: tauri::AppHandle, port: Option<u32>) -> Result<Option<u16>, String> {
    let port = validate_port(port)?;
    write_preference(&app_config_dir(&app)?, port)?;
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_round_trips_and_invalid_data_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_preference(dir.path()), None);

        write_preference(dir.path(), Some(43210)).unwrap();
        assert_eq!(read_preference(dir.path()), Some(43210));

        fs::write(preference_path(dir.path()), "70000\n").unwrap();
        assert_eq!(read_preference(dir.path()), None);
        assert!(validate_port(Some(0)).is_err());
        assert!(validate_port(Some(65536)).is_err());
    }

    #[test]
    fn automatic_allows_fallback_and_custom_is_exact() {
        assert_eq!(
            policy_for(None),
            LaunchPolicy {
                port: AUTOMATIC_PORT,
                exact: false,
            }
        );
        assert_eq!(
            policy_for(Some(43210)),
            LaunchPolicy {
                port: 43210,
                exact: true,
            }
        );
    }
}
