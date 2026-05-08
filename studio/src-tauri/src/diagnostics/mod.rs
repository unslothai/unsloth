mod phase_log;
mod redaction;
mod report;
mod state;

use crate::process::BackendState;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::AppHandle;

pub use phase_log::append_phase_line;
#[cfg(target_os = "linux")]
pub use phase_log::PhaseLogHandle;
pub use state::{
    begin_adopted_backend_session, begin_backend_session, begin_install_attempt,
    begin_repair_child, begin_repair_group, begin_update_attempt, finish_attempt,
    finish_repair_group, new_diagnostics_state, record_attached_external_backend,
    record_auth_failure, record_backend_exit, record_backend_intentional_stop, record_backend_port,
    record_backend_start_failure, record_backend_watchdog, record_diag_marker,
    record_elevation_packages, record_preflight, record_progress, record_step, AttemptLog,
    BackendLog, DiagnosticsState, FrontendSupportSnapshot,
};

pub const SCHEMA_VERSION: u32 = 1;
pub const PHASE_LOG_SEGMENT_MAX_BYTES: u64 = 5 * 1024 * 1024;
pub const PHASE_LOG_MAX_SEGMENTS_PER_GROUP: usize = 3;
#[allow(dead_code)]
pub const PHASE_LOG_KEEP_GROUPS_PER_KIND: usize = 5;
pub const TAIL_MAX_LINES: usize = 1000;
pub const TAIL_MAX_BYTES: usize = 200 * 1024;
pub const REPORT_MAX_BYTES: usize = 1024 * 1024;

pub(crate) const MAX_STATE_ITEMS: usize = 200;
pub(crate) const MAX_PHASE_LINE_BYTES: usize = 16 * 1024;
pub(crate) const FOOTER_BUDGET_BYTES: usize = 8 * 1024;

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

pub(crate) fn new_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}-{}", sanitize_component(prefix), now_ms(), n)
}

pub fn studio_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".unsloth")
        .join("studio")
}

pub fn logs_dir() -> PathBuf {
    studio_dir().join("logs")
}

pub(crate) fn cap_string(value: &str, max: usize) -> String {
    if value.len() <= max {
        return value.to_string();
    }
    let boundary = valid_utf8_boundary(value, max);
    format!("{} [truncated]", &value[..boundary])
}

pub(crate) fn valid_utf8_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut index = max;
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub(crate) fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[tauri::command]
pub async fn collect_support_diagnostics(
    _app: AppHandle,
    diagnostics: tauri::State<'_, DiagnosticsState>,
    backend: tauri::State<'_, BackendState>,
    snapshot: FrontendSupportSnapshot,
) -> Result<String, String> {
    let diagnostics_snapshot = state::clone_snapshot(diagnostics.inner());
    let backend_port = backend.lock().ok().and_then(|proc| proc.port).or_else(|| {
        diagnostics_snapshot
            .backend
            .as_ref()
            .and_then(|backend| backend.reported_port.or(backend.requested_port))
    });

    let health = match backend_port {
        Some(port) => Some(collect_backend_health(port).await),
        None => None,
    };

    match tokio::task::spawn_blocking(move || {
        report::render_report(diagnostics_snapshot, snapshot, health)
    })
    .await
    {
        Ok(report) => Ok(report),
        Err(error) => Ok(format!(
            "DEGRADED RUST DIAGNOSTICS\ncollection_warning=report task failed: {error}\n"
        )),
    }
}

async fn collect_backend_health(port: u16) -> report::BackendHealthSection {
    let mut section = report::BackendHealthSection {
        port,
        fields: Vec::new(),
        warning: None,
    };
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(750))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            section.warning = Some(format!("health client unavailable: {error}"));
            return section;
        }
    };
    let url = format!("http://127.0.0.1:{port}/api/health");
    match client.get(url).send().await {
        Ok(response) => match response.json::<serde_json::Value>().await {
            Ok(json) => {
                for key in [
                    "status",
                    "service",
                    "version",
                    "device_type",
                    "chat_only",
                    "desktop_protocol_version",
                    "desktop_manageability_version",
                    "supports_api_only",
                    "supports_desktop_auth",
                    "supports_desktop_backend_ownership",
                ] {
                    if let Some(value) = json.get(key) {
                        section
                            .fields
                            .push((key.to_string(), report::selected_json_value(value)));
                    }
                }
            }
            Err(error) => section.warning = Some(format!("health json unavailable: {error}")),
        },
        Err(error) => section.warning = Some(format!("health request unavailable: {error}")),
    }
    section
}
