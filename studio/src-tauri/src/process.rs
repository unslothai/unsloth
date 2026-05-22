use crate::diagnostics::{self, BackendLog, DiagnosticsState};
use log::{error, info, warn};
use process_wrap::std::*;
use regex::Regex;
use std::collections::VecDeque;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

const MAX_LOG_LINES: usize = 1000;

#[allow(dead_code)]
pub(crate) enum OwnedBackendHandle {
    Spawned {
        child: Box<dyn ChildWrapper + Send>,
        owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
        reported_port: Option<u16>,
        pid: u32,
        generation: u64,
    },
    Adopted {
        owner: crate::desktop_backend_owner::BackendOwnerState,
        port: u16,
        pid: u32,
        generation: u64,
    },
}

#[allow(dead_code)]
impl OwnedBackendHandle {
    pub(crate) fn spawned(
        child: Box<dyn ChildWrapper + Send>,
        owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
        pid: u32,
        generation: u64,
    ) -> Self {
        Self::Spawned {
            child,
            owner,
            reported_port: None,
            pid,
            generation,
        }
    }

    pub(crate) fn adopted(
        owner: crate::desktop_backend_owner::BackendOwnerState,
        port: u16,
        pid: u32,
        generation: u64,
    ) -> Self {
        Self::Adopted {
            owner,
            port,
            pid,
            generation,
        }
    }

    pub(crate) fn port(&self) -> Option<u16> {
        match self {
            Self::Spawned { reported_port, .. } => *reported_port,
            Self::Adopted { port, .. } => Some(*port),
        }
    }

    fn set_reported_port(&mut self, port: u16) {
        if let Self::Spawned {
            reported_port,
            owner,
            ..
        } = self
        {
            *reported_port = Some(port);
            if let Some(owner) = owner.as_mut() {
                if let Err(error) = owner.update_port(port) {
                    warn!("Could not update desktop backend owner metadata: {}", error);
                }
            }
        }
    }

    fn spawned_child_mut(&mut self) -> Option<&mut Box<dyn ChildWrapper + Send>> {
        match self {
            Self::Spawned { child, .. } => Some(child),
            Self::Adopted { .. } => None,
        }
    }

    fn remove_owner_metadata(self) {
        match self {
            Self::Spawned {
                owner: Some(owner), ..
            }
            | Self::Adopted { owner, .. } => owner.remove(),
            Self::Spawned { owner: None, .. } => {}
        }
    }
}

pub struct BackendProcess {
    pub owned: Option<OwnedBackendHandle>,
    pub port: Option<u16>,
    pub logs: VecDeque<String>,
    pub intentional_stop: bool,
    pub generation: u64,
    pub diagnostics_session: Option<BackendLog>,
    pub adopted_watchdog_generation: Option<u64>,
}

impl BackendProcess {
    pub(crate) fn has_owned_backend(&self) -> bool {
        self.owned.is_some()
    }

    pub(crate) fn has_adopted_backend(&self) -> bool {
        matches!(self.owned, Some(OwnedBackendHandle::Adopted { .. }))
    }

    pub(crate) fn owned_backend_port(&self) -> Option<u16> {
        self.owned.as_ref().and_then(OwnedBackendHandle::port)
    }
}

#[derive(Clone)]
pub(crate) struct OwnedBackendSnapshot {
    pub(crate) owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
    pub(crate) port: Option<u16>,
    pub(crate) generation: u64,
    pub(crate) is_adopted: bool,
}

pub(crate) struct AdoptedBackendState {
    pub(crate) generation: u64,
    pub(crate) newly_adopted: bool,
}

pub(crate) fn adopt_verified_backend(
    state: &BackendState,
    verified: crate::desktop_backend_owner::VerifiedOwnedBackend,
) -> Result<AdoptedBackendState, String> {
    let mut proc = state.lock().map_err(|e| e.to_string())?;
    if proc.has_owned_backend() {
        if proc.owned_backend_port() == Some(verified.port) {
            proc.port = Some(verified.port);
            return Ok(AdoptedBackendState {
                generation: proc.generation,
                newly_adopted: false,
            });
        }
        return Err("Backend is already running.".to_string());
    }

    proc.generation = proc.generation.wrapping_add(1);
    proc.port = Some(verified.port);
    proc.logs.clear();
    proc.intentional_stop = false;
    proc.diagnostics_session = None;
    proc.adopted_watchdog_generation = None;
    proc.owned = Some(OwnedBackendHandle::adopted(
        verified.owner,
        verified.port,
        verified.backend_pid,
        verified.generation,
    ));
    Ok(AdoptedBackendState {
        generation: proc.generation,
        newly_adopted: true,
    })
}

pub(crate) fn owned_backend_snapshot(
    state: &BackendState,
) -> Result<Option<OwnedBackendSnapshot>, String> {
    let proc = state.lock().map_err(|e| e.to_string())?;
    let snapshot = match proc.owned.as_ref() {
        Some(OwnedBackendHandle::Spawned {
            owner,
            reported_port,
            ..
        }) => Some(OwnedBackendSnapshot {
            owner: owner.clone(),
            port: *reported_port,
            generation: proc.generation,
            is_adopted: false,
        }),
        Some(OwnedBackendHandle::Adopted { owner, port, .. }) => Some(OwnedBackendSnapshot {
            owner: Some(owner.clone()),
            port: Some(*port),
            generation: proc.generation,
            is_adopted: true,
        }),
        None => None,
    };
    Ok(snapshot)
}

pub(crate) fn record_owned_backend_port_if_current(
    state: &BackendState,
    generation: u64,
    port: u16,
) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation {
        return false;
    }
    match proc.owned.as_mut() {
        Some(OwnedBackendHandle::Spawned { .. }) => {
            proc.port = Some(port);
            if let Some(owned) = proc.owned.as_mut() {
                owned.set_reported_port(port);
            }
            true
        }
        Some(OwnedBackendHandle::Adopted {
            port: current_port, ..
        }) if *current_port == port => {
            proc.port = Some(port);
            true
        }
        _ => false,
    }
}

pub(crate) fn clear_adopted_backend_if_current(
    state: &BackendState,
    generation: u64,
    port: Option<u16>,
    reason: &str,
) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation {
        return false;
    }
    let matches_adopted = matches!(
        proc.owned.as_ref(),
        Some(OwnedBackendHandle::Adopted { port: current_port, .. })
            if port.map_or(true, |port| port == *current_port)
    );
    if !matches_adopted {
        return false;
    }

    warn!("Clearing adopted backend state after {reason}");
    proc.owned = None;
    proc.port = None;
    proc.diagnostics_session = None;
    proc.adopted_watchdog_generation = None;
    true
}

pub(crate) fn claim_adopted_watchdog_if_current(state: &BackendState, generation: u64) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation || !proc.has_adopted_backend() {
        return false;
    }
    if proc.adopted_watchdog_generation == Some(generation) {
        return false;
    }
    proc.adopted_watchdog_generation = Some(generation);
    true
}

pub(crate) fn clear_adopted_watchdog_if_current(state: &BackendState, generation: u64) {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation == generation && proc.adopted_watchdog_generation == Some(generation) {
        proc.adopted_watchdog_generation = None;
    }
}

impl Default for BackendProcess {
    fn default() -> Self {
        Self {
            owned: None,
            port: None,
            logs: VecDeque::with_capacity(MAX_LOG_LINES),
            intentional_stop: false,
            generation: 0,
            diagnostics_session: None,
            adopted_watchdog_generation: None,
        }
    }
}

pub type BackendState = Arc<Mutex<BackendProcess>>;
pub type ShutdownFlag = Arc<AtomicBool>;

pub fn new_backend_state() -> BackendState {
    Arc::new(Mutex::new(BackendProcess::default()))
}

pub fn new_shutdown_flag() -> ShutdownFlag {
    Arc::new(AtomicBool::new(false))
}

pub(crate) fn trim_line_endings(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &bytes[..end]
}

/// Windows `CREATE_NO_WINDOW` flag — suppresses console windows for child processes.
#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Force-kill a Windows process tree via hidden `taskkill /T /F`, falling
/// back to `child.kill()` if taskkill itself fails.  Reaps the child afterward.
#[cfg(windows)]
pub(crate) fn force_kill_process_tree(
    pid: u32,
    child: &mut Box<dyn ChildWrapper + Send>,
    label: &str,
) {
    use std::os::windows::process::CommandExt;

    let taskkill_status = Command::new("taskkill.exe")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match taskkill_status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            warn!(
                "taskkill returned non-zero status for {} pid {}: {}",
                label, pid, status
            );
            let _ = child.kill();
        }
        Err(e) => {
            warn!("taskkill failed for {} pid {}: {}", label, pid, e);
            let _ = child.kill();
        }
    }

    let _ = child.wait();
    info!("{} process tree force stopped", label);
}

/// Returns the path to the unsloth binary inside the managed venv, if it exists.
/// Checks the new layout (~/.unsloth/studio/unsloth_studio/) first,
/// then falls back to the old layout (~/.unsloth/studio/.venv/) for compat.
fn find_unsloth_binary_in_studio_dir(studio: &std::path::Path) -> Option<std::path::PathBuf> {
    // New layout (upstream scripts >= March 2026)
    let new_base = studio.join("unsloth_studio");
    // Old layout (bundled scripts, older upstream)
    let old_base = studio.join(".venv");

    for base in [new_base, old_base] {
        #[cfg(unix)]
        let bin = base.join("bin").join("unsloth");
        #[cfg(windows)]
        let bin = base.join("Scripts").join("unsloth.exe");

        if bin.exists() {
            return Some(bin);
        }
    }

    None
}

pub fn find_unsloth_binary() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let studio = home.join(".unsloth").join("studio");

    find_unsloth_binary_in_studio_dir(&studio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_studio_dir(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn finds_new_layout_before_legacy_layout_and_falls_back() {
        let temp = temp_studio_dir("layout-preference");

        #[cfg(unix)]
        let new_bin = temp.join("unsloth_studio/bin/unsloth");
        #[cfg(unix)]
        let old_bin = temp.join(".venv/bin/unsloth");
        #[cfg(windows)]
        let new_bin = temp.join("unsloth_studio/Scripts/unsloth.exe");
        #[cfg(windows)]
        let old_bin = temp.join(".venv/Scripts/unsloth.exe");

        fs::create_dir_all(new_bin.parent().unwrap()).unwrap();
        fs::create_dir_all(old_bin.parent().unwrap()).unwrap();
        fs::write(&new_bin, "").unwrap();
        fs::write(&old_bin, "").unwrap();

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&temp),
            Some(new_bin.clone())
        );
        fs::remove_file(&new_bin).unwrap();
        assert_eq!(find_unsloth_binary_in_studio_dir(&temp), Some(old_bin));
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn backend_args_always_enable_api_only() {
        assert_eq!(
            backend_args(8888),
            vec!["studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"]
        );
    }

    fn listening_non_studio_port() -> (u16, mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || loop {
            if rx.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut buf = [0_u8; 512];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        });
        (port, tx, handle)
    }

    #[test]
    fn stop_backend_rolls_back_shutdown_flag_when_adopted_stop_fails() {
        let (port, stop_listener, listener_thread) = listening_non_studio_port();
        let state = new_backend_state();
        let shutdown = new_shutdown_flag();
        let owner = crate::desktop_backend_owner::test_owner_state(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "desktop-owner-token",
            port,
        );

        {
            let mut proc = state.lock().unwrap();
            proc.generation = 7;
            proc.port = Some(port);
            proc.owned = Some(OwnedBackendHandle::adopted(owner, port, 1234, 3));
        }

        let error = stop_backend(&state, &shutdown, None)
            .expect_err("adopted backend should refuse unsafe stop fallback");

        assert!(error.contains("Refusing to stop adopted backend"));
        assert!(!shutdown.load(Ordering::SeqCst));
        assert!(state.lock().unwrap().has_adopted_backend());

        let _ = stop_listener.send(());
        let _ = std::net::TcpStream::connect(("127.0.0.1", port));
        listener_thread.join().unwrap();
    }
}

/// Find the unsloth binary, preferring the dev repo if available.
/// In dev mode (debug builds), checks for a local .venv in the repo first.
/// Falls back to find_unsloth_binary() which checks ~/.unsloth/studio/unsloth_studio/
/// (new layout) then ~/.unsloth/studio/.venv/ (old layout).
pub(crate) fn resolve_backend_binary() -> Result<std::path::PathBuf, String> {
    // In dev mode, check for local repo venv first
    #[cfg(debug_assertions)]
    {
        // CARGO_MANIFEST_DIR is set at compile time to studio/src-tauri/
        // Repo root is 2 levels up: studio/src-tauri -> studio -> repo_root
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let repo_root = std::path::Path::new(manifest_dir)
            .parent() // studio/
            .and_then(|p| p.parent()); // repo_root/

        if let Some(root) = repo_root {
            #[cfg(unix)]
            let dev_bin = root.join(".venv/bin/unsloth");
            #[cfg(windows)]
            let dev_bin = root.join(".venv/Scripts/unsloth.exe");

            if dev_bin.exists() {
                info!("Dev mode: using local repo backend at {:?}", dev_bin);
                return Ok(dev_bin.to_path_buf());
            }
        }
        info!("Dev mode: no local .venv found, falling back to installed backend");
    }

    find_unsloth_binary()
        .ok_or_else(|| "Unsloth binary not found. Please install Unsloth Studio first.".to_string())
}

fn backend_args(port: u16) -> Vec<String> {
    [
        "studio",
        "--api-only",
        "-H",
        "127.0.0.1",
        "-p",
        &port.to_string(),
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Spawn the backend process and wire up stdout/stderr reader threads.
pub fn start_backend(
    app: &AppHandle,
    state: &BackendState,
    port: u16,
    shutdown: &ShutdownFlag,
    diagnostics_state: &DiagnosticsState,
) -> Result<u64, String> {
    let bin = match resolve_backend_binary() {
        Ok(bin) => bin,
        Err(msg) => {
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "resolve_backend_binary",
                &msg,
            );
            return Err(msg);
        }
    };

    let args = backend_args(port);
    let start_line = format!("Starting backend: {:?} {}", bin, args.join(" "));
    let pending_owner = crate::desktop_backend_owner::new_pending_owner();
    let mut cmd = Command::new(&bin);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(native_state) = app.try_state::<crate::native_intents::NativeIntakeState>() {
        cmd.env(
            crate::native_backend_lease::LEASE_SECRET_ENV,
            native_state.lease_secret_env(),
        );
    }

    if let Some(owner) = pending_owner.as_ref() {
        cmd.env(
            crate::desktop_backend_owner::OWNER_TOKEN_ENV,
            owner.token.as_str(),
        );
        cmd.env(
            crate::desktop_backend_owner::OWNER_KIND_ENV,
            crate::desktop_backend_owner::OWNER_KIND_TAURI,
        );
    }

    // AppImage sets LD_LIBRARY_PATH to its bundled libs, which breaks the spawned
    // Python process (wrong libpython/libz → "No module named encodings").
    // Only clear when running inside an AppImage — native package installs may
    // need these env vars for custom CUDA or conda paths.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // scrub so the spawned Python backend can't diverge. UNSLOTH_LLAMA_CPP_PATH
    // is a pre-existing user-controlled llama.cpp dir override; keep it.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    // Reset state, spawn, and store the child while holding the backend mutex.
    // This keeps the no-child check atomic: a concurrent start/stop cannot slip
    // into the old window between generation reset and child storage.
    let (generation, backend_log, stdout, stderr) = {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        if proc.has_owned_backend() {
            return Err("Backend is already running.".to_string());
        }

        shutdown.store(false, Ordering::SeqCst);
        proc.generation = proc.generation.wrapping_add(1);
        proc.port = None;
        proc.logs.clear();
        proc.intentional_stop = false;
        proc.diagnostics_session = None;
        proc.adopted_watchdog_generation = None;
        proc.owned = None;
        let generation = proc.generation;

        let backend_log = diagnostics::begin_backend_session(diagnostics_state, port, generation);

        // On Windows, launch the backend directly with hidden-window flags.
        // The app process is assigned to a KILL_ON_JOB_CLOSE job in main.rs, so
        // children inherit crash-safe cleanup without the buggy per-child JobObject wrapper.
        #[cfg(windows)]
        let mut child: Box<dyn ChildWrapper + Send> = {
            use std::os::windows::process::CommandExt;

            const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
            let child = cmd.spawn().map_err(|e| {
                let msg = format!("Failed to spawn backend: {}", e);
                diagnostics::record_backend_start_failure(
                    diagnostics_state,
                    Some(port),
                    Some(generation),
                    "spawn_backend",
                    &msg,
                );
                msg
            })?;
            Box::new(child)
        };

        #[cfg(unix)]
        let mut child: Box<dyn ChildWrapper + Send> = {
            // Keep the backend tree in a process group on Unix for cleanup.
            let mut wrap = CommandWrap::from(cmd);
            wrap.wrap(ProcessGroup::leader());
            wrap.spawn().map_err(|e| {
                let msg = format!("Failed to spawn backend: {}", e);
                diagnostics::record_backend_start_failure(
                    diagnostics_state,
                    Some(port),
                    Some(generation),
                    "spawn_backend",
                    &msg,
                );
                msg
            })?
        };

        let backend_pid = child.id();
        let stdout = child.stdout().take();
        let stderr = child.stderr().take();
        let owner = pending_owner.clone().and_then(|pending| {
            crate::desktop_backend_owner::activate_owner(pending, port, generation, backend_pid)
        });

        proc.owned = Some(OwnedBackendHandle::spawned(
            child,
            owner,
            backend_pid,
            generation,
        ));
        proc.diagnostics_session = Some(backend_log.clone());
        (generation, backend_log, stdout, stderr)
    };

    info!("{}", start_line);
    diagnostics::append_phase_line(&backend_log.handle, "meta", &start_line);

    if let Some(stdout) = stdout {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stdout,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                false,
                generation,
            );
        });
    }

    if let Some(stderr) = stderr {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stderr,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                true,
                generation,
            );
        });
    }

    Ok(generation)
}

async fn generic_backend_health_ok(port: u16) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            warn!("Could not build backend validation client: {}", error);
            return false;
        }
    };
    let response = match client
        .get(format!("http://127.0.0.1:{port}/api/health"))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            warn!(
                "Backend port candidate {} failed health request: {}",
                port, error
            );
            return false;
        }
    };
    if !response.status().is_success() {
        warn!(
            "Backend port candidate {} returned HTTP {} from health",
            port,
            response.status()
        );
        return false;
    }
    let json = match response.json::<serde_json::Value>().await {
        Ok(json) => json,
        Err(error) => {
            warn!(
                "Backend port candidate {} returned invalid health JSON: {}",
                port, error
            );
            return false;
        }
    };
    let healthy = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "healthy")
        .unwrap_or(false);
    let service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);
    healthy && service
}

async fn validate_candidate_port(
    app: AppHandle,
    state: BackendState,
    diagnostics_state: DiagnosticsState,
    session_id: String,
    generation: u64,
    port: u16,
) {
    let owner = {
        let proc = match state.lock() {
            Ok(proc) => proc,
            Err(error) => {
                warn!("Backend state unavailable for port validation: {}", error);
                return;
            }
        };
        if proc.generation != generation || proc.port.is_some() {
            return;
        }
        match proc.owned.as_ref() {
            Some(OwnedBackendHandle::Spawned { owner, .. }) => owner.clone(),
            _ => return,
        }
    };

    let valid = if let Some(owner) = owner {
        matches!(
            crate::desktop_backend_owner::probe_owned_backend_state(owner, Some(port), false).await,
            crate::desktop_backend_owner::OwnedBackendProbe::Verified(
                crate::desktop_backend_owner::VerifiedOwnedBackend { port: verified_port, .. }
            ) if verified_port == port
        )
    } else {
        generic_backend_health_ok(port).await
    };

    if !valid {
        warn!("Ignoring unverified TAURI_PORT candidate {}", port);
        return;
    }

    let should_emit = {
        let mut proc = match state.lock() {
            Ok(proc) => proc,
            Err(error) => {
                warn!("Backend state unavailable after port validation: {}", error);
                return;
            }
        };
        if proc.generation != generation || proc.port.is_some() {
            false
        } else if matches!(proc.owned, Some(OwnedBackendHandle::Spawned { .. })) {
            proc.port = Some(port);
            if let Some(owned) = proc.owned.as_mut() {
                owned.set_reported_port(port);
            }
            true
        } else {
            false
        }
    };

    if should_emit {
        diagnostics::record_backend_port(&diagnostics_state, &session_id, port);
        info!("Validated backend port: {}", port);
        let _ = app.emit("server-port", port);
    }
}

/// Read lines from a child process stream (stdout or stderr).
/// For stdout, parse TAURI_PORT=(\d+) candidates for async validation.
/// When stdout closes and the stop was not intentional, emit server-crashed.
fn read_output_stream<R: std::io::Read>(
    stream: R,
    app: &AppHandle,
    state: &BackendState,
    diagnostics_state: &DiagnosticsState,
    backend_log: &BackendLog,
    is_stderr: bool,
    generation: u64,
) {
    let mut reader = std::io::BufReader::new(stream);
    let port_re = Regex::new(r"TAURI_PORT=(\d+)").unwrap();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
                let log_line = if is_stderr {
                    format!("[stderr] {}", text)
                } else {
                    text.clone()
                };

                diagnostics::append_phase_line(
                    &backend_log.handle,
                    if is_stderr { "stderr" } else { "stdout" },
                    &text,
                );

                let detected_port = if !is_stderr {
                    port_re
                        .captures(&text)
                        .and_then(|caps| caps.get(1))
                        .and_then(|port_str| port_str.as_str().parse::<u16>().ok())
                } else {
                    None
                };

                // Buffer the log line only for the current backend generation.
                // Old reader threads can briefly outlive a stop/start cycle;
                // they must not overwrite the new backend's port or logs.
                let mut candidate_port = None;
                let current_generation = if let Ok(mut proc) = state.lock() {
                    if proc.generation != generation {
                        false
                    } else {
                        candidate_port = detected_port;
                        if proc.logs.len() >= MAX_LOG_LINES {
                            proc.logs.pop_front();
                        }
                        proc.logs.push_back(log_line.clone());
                        true
                    }
                } else {
                    false
                };

                if !current_generation {
                    break;
                }

                if let Some(port) = candidate_port {
                    let app_handle = app.clone();
                    let state_clone = Arc::clone(state);
                    let diagnostics_clone = diagnostics_state.clone();
                    let session_id = backend_log.session_id.clone();
                    tauri::async_runtime::spawn(async move {
                        validate_candidate_port(
                            app_handle,
                            state_clone,
                            diagnostics_clone,
                            session_id,
                            generation,
                            port,
                        )
                        .await;
                    });
                }

                info!("[backend] {}", log_line);

                let _ = app.emit("server-log", &log_line);
            }
            Err(e) => {
                warn!(
                    "Error reading backend {}: {}",
                    if is_stderr { "stderr" } else { "stdout" },
                    e
                );
                break;
            }
        }
    }

    // Stream closed. Only the stdout reader checks for crashes.
    if !is_stderr {
        let mut exit_record: Option<(String, bool)> = None;
        let mut emit_crash = false;
        if let Ok(mut proc) = state.lock() {
            if proc.generation != generation {
                return;
            }
            let intentional = proc.intentional_stop;
            let exited = if let Some(child) = proc
                .owned
                .as_mut()
                .and_then(OwnedBackendHandle::spawned_child_mut)
            {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        info!("Backend stdout stream ended with status: {}", status);
                        exit_record = Some((status.to_string(), intentional));
                        true
                    }
                    Ok(None) => {
                        warn!("Backend stdout stream ended, but process is still running");
                        false
                    }
                    Err(e) => {
                        warn!("Failed to query backend status after stdout closed: {}", e);
                        false
                    }
                }
            } else {
                false
            };

            if exited {
                if let Some(owned) = proc.owned.take() {
                    owned.remove_owner_metadata();
                }
                proc.port = None;
                proc.diagnostics_session = None;
                emit_crash = !intentional;
            }
        }
        if let Some((status, intentional)) = exit_record {
            diagnostics::record_backend_exit(
                diagnostics_state,
                &backend_log.session_id,
                Some(status),
                intentional,
                None,
            );
        }
        if emit_crash {
            error!("Backend process stdout closed unexpectedly (crash detected)");
            let _ = app.emit("server-crashed", ());
        }
    }
}

fn wait_for_child_exit(child: &mut Box<dyn ChildWrapper + Send>, label: &str) -> bool {
    for _ in 0..50 {
        match child.try_wait() {
            Ok(Some(status)) => {
                info!("{} exited with status: {}", label, status);
                return true;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => {
                warn!("Error polling {} process: {}", label, e);
                return false;
            }
        }
    }
    false
}

fn wait_for_port_disconnect(port: u16, timeout: Duration) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < timeout {
        if !crate::desktop_backend_owner::port_is_listening_blocking(
            port,
            Duration::from_millis(150),
        ) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn try_exact_port_http_shutdown(port: u16, label: &str) -> bool {
    match crate::desktop_backend_owner::exact_port_http_shutdown_blocking(port) {
        Ok(()) => {
            info!(
                "{} exact-port HTTP shutdown requested on port {}",
                label, port
            );
            true
        }
        Err(error) => {
            warn!(
                "{} exact-port HTTP shutdown failed on port {}: {}",
                label, port, error
            );
            false
        }
    }
}

fn remove_optional_owner(owner: Option<crate::desktop_backend_owner::BackendOwnerState>) {
    if let Some(owner) = owner {
        owner.remove();
    }
}

fn stop_spawned_backend(
    mut child: Box<dyn ChildWrapper + Send>,
    owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
    reported_port: Option<u16>,
    pid: u32,
) -> Result<(), String> {
    #[cfg(not(windows))]
    let _ = reported_port;
    info!("Stopping spawned backend process group (pid {})", pid);

    #[cfg(windows)]
    if let Some(port) = reported_port {
        let verified = owner
            .as_ref()
            .map(|owner| owner.verifies_exact_port_blocking(port))
            .unwrap_or(false);
        if verified
            && try_exact_port_http_shutdown(port, "Spawned backend")
            && wait_for_child_exit(&mut child, "Backend")
        {
            remove_optional_owner(owner);
            return Ok(());
        }
    }

    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            warn!("PID {} exceeds i32 range, using direct kill", pid);
            let _ = child.kill();
            let _ = child.wait();
            remove_optional_owner(owner);
            return Ok(());
        }
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
    }

    #[cfg(windows)]
    {
        unsafe {
            windows_sys::Win32::System::Console::GenerateConsoleCtrlEvent(
                windows_sys::Win32::System::Console::CTRL_BREAK_EVENT,
                pid,
            );
        }
    }

    if wait_for_child_exit(&mut child, "Backend") {
        remove_optional_owner(owner);
        return Ok(());
    }

    #[cfg(windows)]
    {
        warn!(
            "Backend did not exit gracefully, force killing process tree (pid {})",
            pid
        );
        force_kill_process_tree(pid, &mut child, "Backend");
        remove_optional_owner(owner);
        return Ok(());
    }

    #[cfg(unix)]
    {
        warn!(
            "Backend did not exit gracefully, force killing group (pid {})",
            pid
        );
        let _ = child.kill();
        let _ = child.wait();
        remove_optional_owner(owner);
        info!("Backend process group forcefully stopped");
        Ok(())
    }
}

fn stop_adopted_backend(
    owner: crate::desktop_backend_owner::BackendOwnerState,
    port: u16,
    pid: u32,
) -> Result<(), String> {
    info!(
        "Stopping adopted desktop-owned backend on exact port {} (pid {})",
        port, pid
    );

    if !owner.verifies_exact_port_blocking(port) {
        return Err(
            "Refusing to stop adopted backend because ownership could not be verified".to_string(),
        );
    }

    if try_exact_port_http_shutdown(port, "Adopted backend")
        && wait_for_port_disconnect(port, Duration::from_secs(5))
    {
        owner.remove();
        return Ok(());
    }

    Err(
        "Adopted backend did not stop via exact-port HTTP shutdown; refusing PID fallback without verified port-to-PID binding"
            .to_string(),
    )
}

/// Graceful shutdown of owned backend handles.
/// Unix spawned: SIGTERM to process group -> wait -> SIGKILL.
/// Windows spawned: exact-port HTTP shutdown -> CTRL_BREAK_EVENT -> taskkill.
/// Adopted handles: exact-port HTTP shutdown only; PID fallback is refused
/// until the backend process identity can be bound to the verified port.
pub fn stop_backend(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
) -> Result<(), String> {
    stop_backend_inner(state, shutdown, diagnostics_state, true)
}

/// Stop before update/repair mutations without letting the watchdog exit unless the stop succeeds.
pub fn stop_backend_for_mutation(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
) -> Result<(), String> {
    stop_backend_inner(state, shutdown, diagnostics_state, false)
}

fn stop_backend_inner(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
    signal_shutdown_before_stop: bool,
) -> Result<(), String> {
    let previous_shutdown = shutdown.load(Ordering::SeqCst);
    if signal_shutdown_before_stop {
        shutdown.store(true, Ordering::SeqCst);
    }
    if let Some(diagnostics_state) = diagnostics_state {
        diagnostics::record_backend_intentional_stop(diagnostics_state);
    }

    enum StopTarget {
        Spawned(OwnedBackendHandle),
        Adopted {
            owner: crate::desktop_backend_owner::BackendOwnerState,
            port: u16,
            pid: u32,
            generation: u64,
            local_generation: u64,
        },
    }

    let target = {
        let mut proc = match state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Backend state mutex poisoned, recovering for cleanup");
                poisoned.into_inner()
            }
        };
        proc.intentional_stop = true;
        match proc.owned.as_ref() {
            Some(OwnedBackendHandle::Spawned { .. }) => {
                proc.port = None;
                proc.diagnostics_session = None;
                proc.adopted_watchdog_generation = None;
                proc.owned.take().map(StopTarget::Spawned)
            }
            Some(OwnedBackendHandle::Adopted {
                owner,
                port,
                pid,
                generation,
            }) => Some(StopTarget::Adopted {
                owner: owner.clone(),
                port: *port,
                pid: *pid,
                generation: *generation,
                local_generation: proc.generation,
            }),
            None => None,
        }
    };

    let result = match target {
        Some(StopTarget::Spawned(OwnedBackendHandle::Spawned {
            child,
            owner,
            reported_port,
            pid,
            ..
        })) => stop_spawned_backend(child, owner, reported_port, pid),
        Some(StopTarget::Adopted {
            owner,
            port,
            pid,
            generation,
            local_generation,
        }) => {
            if let Err(error) = stop_adopted_backend(owner, port, pid) {
                if !crate::desktop_backend_owner::port_is_listening_blocking(
                    port,
                    Duration::from_millis(150),
                ) {
                    clear_adopted_backend_if_current(
                        state,
                        local_generation,
                        Some(port),
                        "adopted port disappeared during stop",
                    );
                    Ok(())
                } else {
                    Err(error)
                }
            } else {
                let mut proc = match state.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        warn!("Backend state mutex poisoned, recovering after adopted stop");
                        poisoned.into_inner()
                    }
                };
                if matches!(
                    proc.owned.as_ref(),
                    Some(OwnedBackendHandle::Adopted {
                        port: current_port,
                        pid: current_pid,
                        generation: current_generation,
                        ..
                    }) if *current_port == port && *current_pid == pid && *current_generation == generation
                ) {
                    proc.owned = None;
                    proc.port = None;
                    proc.diagnostics_session = None;
                    proc.adopted_watchdog_generation = None;
                }
                Ok(())
            }
        }
        Some(StopTarget::Spawned(OwnedBackendHandle::Adopted { .. })) => unreachable!(),
        None => Ok(()),
    };

    if result.is_ok() && !signal_shutdown_before_stop {
        shutdown.store(true, Ordering::SeqCst);
    } else if result.is_err() && signal_shutdown_before_stop {
        shutdown.store(previous_shutdown, Ordering::SeqCst);
    }

    result
}
