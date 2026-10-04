use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::phase_log::{collect_phase_groups, path_has_symlink};
use super::redaction::{redact_text, RedactionReport};
use super::state::{AttemptSummary, CappedStrings, DiagnosticsSnapshot, FrontendSupportSnapshot};
use super::{
    cap_string, logs_dir, now_ms, studio_dir, valid_utf8_boundary, FOOTER_BUDGET_BYTES,
    MAX_PHASE_LINE_BYTES, REPORT_MAX_BYTES, SCHEMA_VERSION, TAIL_MAX_BYTES, TAIL_MAX_LINES,
};

#[derive(Clone, Debug)]
pub(crate) struct BackendHealthSection {
    pub(crate) port: u16,
    pub(crate) fields: Vec<(String, String)>,
    pub(crate) warning: Option<String>,
}

pub(crate) fn render_report(
    snapshot: DiagnosticsSnapshot,
    frontend: FrontendSupportSnapshot,
    health: Option<BackendHealthSection>,
) -> String {
    let mut warnings = Vec::new();
    let mut raw = String::new();

    raw.push_str("Unsloth Support Diagnostics\n");
    raw.push_str(&format!("diag_report_schema={SCHEMA_VERSION}\n"));
    raw.push_str(&format!("created_at_ms={}\n", now_ms()));
    raw.push_str(&format!("app_version={}\n", env!("CARGO_PKG_VERSION")));
    raw.push_str(&format!("os={}\n", std::env::consts::OS));
    raw.push_str(&format!("arch={}\n", std::env::consts::ARCH));
    raw.push_str(&format!("debug_build={}\n", cfg!(debug_assertions)));
    raw.push_str(&format!(
        "appimage={}\n",
        std::env::var_os("APPIMAGE").is_some()
    ));
    raw.push_str(&format!(
        "cpu_logical_count={}\n",
        std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "unavailable".to_string())
    ));
    raw.push_str(&format!("studio_dir={}\n", studio_dir().display()));
    raw.push_str(&format!("logs_dir={}\n", logs_dir().display()));
    raw.push_str(&format!("app_session_id={}\n", snapshot.app_session_id));
    raw.push_str(&format!(
        "app_started_at_ms={}\n",
        snapshot.app_started_at_ms
    ));
    raw.push_str(
        "disk_budget=5MiB x 3 segments x 5 groups x 4 kinds ~= 300MiB plus tauri.log/.1\n",
    );

    append_frontend_section(&mut raw, &frontend);
    append_preflight_section(&mut raw, &snapshot);
    append_flow_section(&mut raw, &snapshot);
    append_backend_section(&mut raw, &snapshot, health.as_ref());
    append_device_signals_section(&mut raw, &snapshot, health.as_ref());
    append_log_sections(&mut raw, &snapshot, &mut warnings);

    if let Some(health) = &health {
        if let Some(warning) = &health.warning {
            warnings.push(format!("backend /api/health unavailable: {warning}"));
        }
    }

    let mut redaction = RedactionReport::default();
    let redacted_body = redact_text(&raw, &mut redaction);

    // Redact collection warnings/footer too. Some warnings include raw local paths
    // from failed log reads, so the footer must not be appended after redaction.
    let footer_for_count = render_footer(&warnings, &redaction);
    let _ = redact_text(&footer_for_count, &mut redaction);
    let footer = redact_text(
        &render_footer(&warnings, &redaction),
        &mut RedactionReport::default(),
    );

    enforce_report_limit(redacted_body, footer)
}

fn append_frontend_section(out: &mut String, frontend: &FrontendSupportSnapshot) {
    out.push_str("\n== Trigger/UI state ==\n");
    append_opt(out, "status", frontend.status.as_deref());
    append_opt(out, "error", frontend.error.as_deref());
    append_opt(
        out,
        "current_step_index",
        frontend
            .current_step_index
            .map(|v| v.to_string())
            .as_deref(),
    );
    append_opt(out, "progress_detail", frontend.progress_detail.as_deref());
    append_opt(out, "flow", frontend.flow.as_deref());
    append_opt(out, "update_phase", frontend.update_phase.as_deref());
    let update_progress = frontend.update_progress.as_ref().map(selected_json_value);
    append_opt(out, "update_progress", update_progress.as_deref());
    if let Some(packages) = &frontend.elevation_packages {
        out.push_str(&format!("elevation_packages={}\n", packages.join(", ")));
    }
    out.push_str("last_ui_log_tail:\n");
    match &frontend.last_ui_log_lines {
        Some(lines) if !lines.is_empty() => {
            for line in tail_lines(lines, TAIL_MAX_LINES) {
                out.push_str("  ");
                out.push_str(line);
                out.push('\n');
            }
        }
        _ => out.push_str("  unavailable\n"),
    }
}

fn append_preflight_section(out: &mut String, snapshot: &DiagnosticsSnapshot) {
    out.push_str("\n== Preflight/install state ==\n");
    match &snapshot.latest_preflight {
        Some(preflight) => {
            out.push_str("source=live-state\n");
            out.push_str(&format!("recorded_at_ms={}\n", preflight.recorded_at_ms));
            out.push_str(&format!("disposition={}\n", preflight.disposition));
            append_opt(out, "reason", preflight.reason.as_deref());
            append_opt(
                out,
                "port",
                preflight.port.map(|p| p.to_string()).as_deref(),
            );
            out.push_str(&format!("can_auto_repair={}\n", preflight.can_auto_repair));
            if let Some(path) = &preflight.managed_bin {
                out.push_str(&format!("managed_bin={}\n", path.display()));
                out.push_str(&format!("managed_bin_exists={}\n", path.exists()));
            } else {
                out.push_str("managed_bin=unavailable\n");
            }
        }
        None => out.push_str("source=live-state unavailable\n"),
    }
}

fn append_flow_section(out: &mut String, snapshot: &DiagnosticsSnapshot) {
    out.push_str("\n== Flow summaries ==\n");
    append_attempt_summary(out, "install", snapshot.install.as_ref(), "live-state");
    if !snapshot.install_history.is_empty() {
        out.push_str("install_history:\n");
        for install in snapshot.install_history.iter().rev().take(4).rev() {
            append_attempt_summary(
                out,
                "  previous_install",
                Some(install),
                "live-state-history",
            );
        }
    }
    append_attempt_summary(out, "update", snapshot.update.as_ref(), "live-state");

    out.push_str("repair_groups:\n");
    if snapshot.repair_groups.is_empty() {
        out.push_str("  unavailable\n");
    } else {
        for group in snapshot.repair_groups.iter().rev().take(5).rev() {
            out.push_str(&format!(
                "  repair_group_id={} started_at_ms={} ended_at_ms={} final_status={} last_error={}\n",
                group.repair_group_id,
                group.started_at_ms,
                opt_u64(group.ended_at_ms),
                group.final_status.as_deref().unwrap_or("unavailable"),
                group.last_error.as_deref().unwrap_or("unavailable")
            ));
            for child in &group.children {
                append_attempt_summary(out, "  child", Some(child), "live-state");
            }
        }
    }
}

fn append_attempt_summary(
    out: &mut String,
    label: &str,
    attempt: Option<&AttemptSummary>,
    source: &str,
) {
    match attempt {
        Some(attempt) => {
            out.push_str(&format!(
                "{label}: source={source} flow={} attempt_id={} group_id={} child_type={} started_at_ms={} ended_at_ms={} exit_status={} intentional_stop={} last_error={} log_segments={}\n",
                attempt.flow,
                attempt.attempt_id,
                attempt.repair_group_id.as_deref().unwrap_or("none"),
                attempt.child_type.as_deref().unwrap_or("none"),
                attempt.started_at_ms,
                opt_u64(attempt.ended_at_ms),
                attempt.exit_status.as_deref().unwrap_or("unavailable"),
                attempt.intentional_stop,
                attempt.last_error.as_deref().unwrap_or("unavailable"),
                attempt
                    .log_segments
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
            if !attempt.elevation_packages.is_empty() {
                out.push_str(&format!(
                    "{label}.elevation_packages={}\n",
                    attempt.elevation_packages.join(", ")
                ));
            }
            append_capped(out, &format!("{label}.steps"), &attempt.steps);
            append_capped(
                out,
                &format!("{label}.progress_details"),
                &attempt.progress_details,
            );
            append_capped(out, &format!("{label}.diag_markers"), &attempt.diag_markers);
        }
        None => out.push_str(&format!("{label}: unavailable\n")),
    }
}

fn append_backend_section(
    out: &mut String,
    snapshot: &DiagnosticsSnapshot,
    health: Option<&BackendHealthSection>,
) {
    out.push_str("\n== Backend/auth/watchdog ==\n");
    match &snapshot.backend {
        Some(backend) => {
            out.push_str("source=live-state\n");
            out.push_str(&format!("backend_kind={}\n", backend.backend_kind));
            append_opt(out, "session_id", backend.session_id.as_deref());
            append_opt(
                out,
                "requested_port",
                backend.requested_port.map(|p| p.to_string()).as_deref(),
            );
            append_opt(
                out,
                "reported_port",
                backend.reported_port.map(|p| p.to_string()).as_deref(),
            );
            append_opt(
                out,
                "generation",
                backend.generation.map(|g| g.to_string()).as_deref(),
            );
            append_opt(
                out,
                "started_at_ms",
                backend.started_at_ms.map(|t| t.to_string()).as_deref(),
            );
            append_opt(
                out,
                "ended_at_ms",
                backend.ended_at_ms.map(|t| t.to_string()).as_deref(),
            );
            append_opt(out, "exit_status", backend.exit_status.as_deref());
            out.push_str(&format!("intentional_stop={}\n", backend.intentional_stop));
            append_opt(out, "terminal_reason", backend.terminal_reason.as_deref());
            append_opt(out, "last_error", backend.last_error.as_deref());
            if backend.backend_kind == "attached_external" {
                out.push_str("managed_process_logs=not_owned_by_tauri\n");
            }
        }
        None => out.push_str("backend=unavailable backend_kind=unknown\n"),
    }

    match &snapshot.auth {
        Some(auth) => {
            out.push_str(&format!(
                "auth_failure: failed_at_ms={} stage={} port={} message={}\n",
                auth.failed_at_ms,
                auth.stage,
                auth.port
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "unavailable".to_string()),
                auth.message
            ));
        }
        None => out.push_str("auth_failure=unavailable\n"),
    }

    out.push_str("runtime_health:\n");
    match health {
        Some(health) => {
            out.push_str(&format!("  port={}\n", health.port));
            if health.fields.is_empty() {
                out.push_str("  selected_fields=unavailable\n");
            } else {
                for (key, value) in &health.fields {
                    out.push_str(&format!("  {key}={value}\n"));
                }
            }
            if let Some(warning) = &health.warning {
                out.push_str(&format!("  warning={warning}\n"));
            }
        }
        None => out.push_str("  unavailable: no known backend port\n"),
    }
}

fn append_device_signals_section(
    out: &mut String,
    snapshot: &DiagnosticsSnapshot,
    health: Option<&BackendHealthSection>,
) {
    out.push_str("\n== Lightweight device signals ==\n");
    out.push_str("scope=setup/runtime signals, not full hardware inventory\n");
    out.push_str(&format!(
        "os={} arch={}\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    out.push_str("installer_diag_markers:\n");
    let mut any = false;
    if let Some(install) = &snapshot.install {
        append_marker_tail(out, "install", install);
        any |= !install.diag_markers.items.is_empty();
    }
    if let Some(update) = &snapshot.update {
        append_marker_tail(out, "update", update);
        any |= !update.diag_markers.items.is_empty();
    }
    for group in &snapshot.repair_groups {
        for child in &group.children {
            append_marker_tail(out, "repair_child", child);
            any |= !child.diag_markers.items.is_empty();
        }
    }
    if !any {
        out.push_str("  unavailable\n");
    }
    if let Some(health) = health {
        out.push_str("runtime_health_selected_fields:\n");
        for (key, value) in &health.fields {
            if matches!(
                key.as_str(),
                "version"
                    | "device_type"
                    | "chat_only"
                    | "desktop_protocol_version"
                    | "desktop_manageability_version"
                    | "supports_api_only"
                    | "supports_desktop_auth"
                    | "supports_desktop_backend_ownership"
            ) {
                out.push_str(&format!("  {key}={value}\n"));
            }
        }
    }
}

fn append_log_sections(
    out: &mut String,
    snapshot: &DiagnosticsSnapshot,
    warnings: &mut Vec<String>,
) {
    out.push_str("\n== Log tails ==\n");
    for name in ["tauri.log", "tauri.log.1"] {
        let path = studio_dir().join(name);
        append_tail_section(out, &path, name, "local-app-log", warnings);
    }

    let server_log_dir = logs_dir().join("server");
    let server_logs = newest_server_logs_in(&server_log_dir, SERVER_LOG_TAIL_FILES, warnings);
    if server_logs.is_empty() {
        // Name the directory: an adopted backend can take its studio root from
        // sys.prefix (storage_roots.py `_infer_studio_home_from_venv`), past any
        // env scrub, so bare "unavailable" cannot be told from "logs elsewhere".
        out.push_str(&format!(
            "backend_session_logs=unavailable searched={}\n",
            server_log_dir.display()
        ));
    }
    for path in server_logs {
        let label = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("server.log")
            .to_string();
        append_tail_section_capped(
            out,
            &path,
            &label,
            "backend-session-log",
            SERVER_LOG_TAIL_MAX_LINES,
            SERVER_LOG_TAIL_MAX_BYTES,
            warnings,
        );
    }

    let phase_groups = collect_phase_groups(snapshot, warnings);
    if phase_groups.is_empty() {
        out.push_str("phase_logs=unavailable\n");
    }
    for group in phase_groups {
        out.push_str(&format!(
            "\n-- phase_log_group label={} source={} --\n",
            group.label, group.source
        ));
        if let Some(note) = &group.note {
            out.push_str(&format!("note={}\n", note));
        }
        for path in group.paths {
            append_tail_section(out, &path, &group.label, &group.source, warnings);
        }
    }
}

/// Two, so a crash and the restart after it both land in one report.
const SERVER_LOG_TAIL_FILES: usize = 2;

/// Smaller than the other tails on purpose: enforce_report_limit chops the END of
/// the body, so bytes spent here come out of the phase logs below, and a faulthandler
/// dump is a few KB at the end of the file, which is the half read_tail returns.
const SERVER_LOG_TAIL_MAX_LINES: usize = 400;
const SERVER_LOG_TAIL_MAX_BYTES: usize = 64 * 1024;

/// Newest `server-*.log` files, most recent first.
///
/// run.py's `_setup_server_disk_logging` aims faulthandler here, so when the GPU
/// runtime aborts this holds the Python stack naming the call that died, and no
/// other file Unsloth keeps does.
///
/// Only two are collected, so an entry is rejected HERE rather than later in
/// `read_tail`: one that merely matches the name would otherwise spend a slot and
/// push the real crash log out. `scan_phase_logs_in_dir` screens during selection
/// for the same reason.
fn newest_server_logs_in(dir: &Path, max: usize, warnings: &mut Vec<String>) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            // Missing is the ordinary case, not a fault: a --no-torch host,
            // UNSLOTH_STUDIO_NO_FILE_LOG=1, or a backend never started. Anything
            // else, a permission wall above all, is worth saying out loud.
            if error.kind() != std::io::ErrorKind::NotFound {
                warnings.push(format!(
                    "backend session log dir unavailable: {} ({})",
                    dir.display(),
                    error
                ));
            }
            return Vec::new();
        }
    };
    let mut logs: Vec<(SystemTime, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !name.starts_with("server-") || !name.ends_with(".log") {
            continue;
        }
        // The header and the path print verbatim into a line-oriented, ```text-fenced
        // report, and a Unix filename may hold a newline or a backtick, so a matching
        // name could forge report structure.
        if name.contains(|ch: char| ch.is_control() || ch == '`') {
            warnings.push(format!(
                "ignored backend session log with an unprintable name in {}",
                dir.display()
            ));
            continue;
        }
        if path_has_symlink(&path) {
            warnings.push(format!(
                "ignored backend session log symlink: {}",
                path.display()
            ));
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        // Regular files only. A directory reads back as EISDIR; a FIFO is worse,
        // since `File::open` on one blocks until a writer appears, hanging report
        // generation rather than failing it.
        if !metadata.is_file() {
            warnings.push(format!(
                "ignored non-regular backend session log: {}",
                path.display()
            ));
            continue;
        }
        // `metadata` is `stat`, which needs no permission on the file itself, so a
        // log we cannot read still looks like a candidate here and would spend a slot
        // that `read_tail` only refuses afterwards. A backend once started under sudo
        // leaves exactly that behind. Safe to open now: non-regular files are already
        // out, so this cannot be the FIFO that blocks.
        if let Err(error) = File::open(&path) {
            warnings.push(format!(
                "ignored unreadable backend session log: {} ({})",
                path.display(),
                error
            ));
            continue;
        }
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        logs.push((modified, path));
    }
    // mtime decides; the name breaks an exact tie, descending like the mtime. `pidN`
    // is not zero-padded, so that tie-break is stable, not chronological.
    logs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    logs.truncate(max);
    logs.into_iter().map(|(_, path)| path).collect()
}

fn append_tail_section(
    out: &mut String,
    path: &Path,
    label: &str,
    source: &str,
    warnings: &mut Vec<String>,
) {
    append_tail_section_capped(
        out,
        path,
        label,
        source,
        TAIL_MAX_LINES,
        TAIL_MAX_BYTES,
        warnings,
    );
}

#[allow(clippy::too_many_arguments)]
fn append_tail_section_capped(
    out: &mut String,
    path: &Path,
    label: &str,
    source: &str,
    max_lines: usize,
    max_bytes: usize,
    warnings: &mut Vec<String>,
) {
    match read_tail(path, max_lines, max_bytes) {
        Ok(tail) => {
            out.push_str(&format!(
                "file={} label={} source={} size_bytes={} mtime_ms={} included_bytes={} included_lines={} truncated={}\n",
                path.display(),
                label,
                source,
                tail.size_bytes,
                opt_u64(tail.mtime_ms),
                tail.included_bytes,
                tail.included_lines,
                tail.truncated
            ));
            out.push_str("```text\n");
            out.push_str(&tail.text);
            if !tail.text.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("```\n");
        }
        Err(error) => {
            out.push_str(&format!(
                "file={} label={} source={} unavailable={}\n",
                path.display(),
                label,
                source,
                error
            ));
            warnings.push(format!("{} unavailable: {}", path.display(), error));
        }
    }
}

#[derive(Debug)]
struct TailRead {
    text: String,
    size_bytes: u64,
    mtime_ms: Option<u64>,
    included_bytes: usize,
    included_lines: usize,
    truncated: bool,
}

fn read_tail(path: &Path, max_lines: usize, max_bytes: usize) -> Result<TailRead, String> {
    if path_has_symlink(path) {
        return Err("refusing to read symlink".to_string());
    }
    let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
    let size = metadata.len();
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let start = size.saturating_sub(max_bytes as u64);
    file.seek(SeekFrom::Start(start))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(max_bytes as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    let mut truncated = start > 0;
    if truncated {
        text = format!("[tail truncated to last {max_bytes} bytes]\n{text}");
    }
    let mut lines: Vec<String> = text
        .lines()
        .map(|line| {
            if line.len() > MAX_PHASE_LINE_BYTES {
                let boundary = valid_utf8_boundary(line, MAX_PHASE_LINE_BYTES);
                format!("{} [line truncated]", &line[..boundary])
            } else {
                line.to_string()
            }
        })
        .collect();
    if lines.len() > max_lines {
        truncated = true;
        let omitted = lines.len() - max_lines;
        lines = lines.split_off(omitted);
        lines.insert(0, format!("[tail truncated to last {max_lines} lines]"));
    }
    let included_lines = lines.len();
    let mut text = lines.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    Ok(TailRead {
        text,
        size_bytes: size,
        mtime_ms: metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64),
        included_bytes: bytes.len(),
        included_lines,
        truncated,
    })
}

fn render_footer(warnings: &[String], redaction: &RedactionReport) -> String {
    let mut footer = String::new();
    footer.push_str("\n== Collection warnings ==\n");
    if warnings.is_empty() {
        footer.push_str("none\n");
    } else {
        for warning in warnings.iter().take(100) {
            footer.push_str("- ");
            footer.push_str(warning);
            footer.push('\n');
        }
        if warnings.len() > 100 {
            footer.push_str(&format!(
                "- omitted {} additional warnings\n",
                warnings.len() - 100
            ));
        }
    }
    footer.push_str("\n== Redaction/omission summary ==\n");
    footer.push_str(&format!(
        "redaction_replacements={}\n",
        redaction.replacements
    ));
    footer.push_str("redaction_scope=ANSI, private keys, URL credentials, auth headers, cookies, token patterns, assignment-style secrets, studio/home paths, emails\n");
    footer.push_str(
        "report_bounds=log sections <=1000 lines/200KiB each; backend session logs <=400 lines/64KiB each; total clipboard text <=1MiB\n",
    );
    footer.push_str("known_v1_gap=elevated apt helper may buffer subprocess output before diagnostics caps it\n");
    footer.push_str("known_v1_gap=normal install elevation resume is linked in same-run state/report history; disk fallback conservatively includes recent install attempts but has no explicit install_group_id in V1\n");
    footer
}

fn enforce_report_limit(body: String, footer: String) -> String {
    if body.len() + footer.len() <= REPORT_MAX_BYTES {
        return format!("{body}{footer}");
    }
    let notice = "\n[report truncated to fit clipboard budget; footer preserved]\n";
    let footer_budget = footer.len().min(FOOTER_BUDGET_BYTES).max(footer.len());
    let budget = REPORT_MAX_BYTES
        .saturating_sub(footer_budget)
        .saturating_sub(notice.len());
    let boundary = valid_utf8_boundary(&body, budget);
    format!("{}{}{}", &body[..boundary], notice, footer)
}
fn append_opt(out: &mut String, key: &str, value: Option<&str>) {
    out.push_str(key);
    out.push('=');
    out.push_str(value.unwrap_or("unavailable"));
    out.push('\n');
}

fn append_capped(out: &mut String, label: &str, values: &CappedStrings) {
    out.push_str(&format!("{label}: omitted={}\n", values.omitted));
    if values.items.is_empty() {
        out.push_str("  unavailable\n");
    } else {
        for value in values.items.iter().rev().take(20).rev() {
            out.push_str("  ");
            out.push_str(value);
            out.push('\n');
        }
    }
}

fn append_marker_tail(out: &mut String, label: &str, attempt: &AttemptSummary) {
    for marker in attempt.diag_markers.items.iter().rev().take(10).rev() {
        out.push_str(&format!("  {label}: {marker}\n"));
    }
}

fn tail_lines(lines: &[String], max_lines: usize) -> &[String] {
    if lines.len() > max_lines {
        &lines[lines.len() - max_lines..]
    } else {
        lines
    }
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

pub(crate) fn selected_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => cap_string(s, 256),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => "null".to_string(),
        _ => "<non-scalar>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_redacts_footer_collection_warning_paths() {
        let snapshot = DiagnosticsSnapshot::default();
        let report = render_report(snapshot, FrontendSupportSnapshot::default(), None);
        let studio = studio_dir().display().to_string();
        assert!(!report.contains(&studio));
        assert!(report.contains("<studio_home>"));
    }

    fn server_log_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-server-logs-{}-{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn collect_names(dir: &Path, max: usize) -> (Vec<String>, Vec<String>) {
        let mut warnings = Vec::new();
        let found = newest_server_logs_in(dir, max, &mut warnings);
        let names = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        (names, warnings)
    }

    #[test]
    fn server_logs_are_newest_first_and_capped() {
        let dir = server_log_dir("order");
        for stamp in ["20260101-000000", "20260102-000000", "20260103-000000"] {
            fs::write(dir.join(format!("server-{stamp}-pid1.log")), b"x").unwrap();
        }

        let (names, _) = collect_names(&dir, 2);
        assert_eq!(
            names,
            vec![
                "server-20260103-000000-pid1.log",
                "server-20260102-000000-pid1.log"
            ]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The test above cannot say WHICH key ordered it, since its names and mtimes
    /// agree. Make them disagree. A copied or restored log carries a stamp that is
    /// not its mtime, and the crash we want is the one written last.
    #[test]
    fn server_logs_order_by_mtime_not_by_name() {
        let dir = server_log_dir("order-key");
        let stale_name = dir.join("server-20260109-000000-pid9.log");
        let fresh_name = dir.join("server-20260101-000000-pid1.log");
        fs::write(&stale_name, b"x").unwrap();
        // Coarser than any filesystem timestamp granularity in play here.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&fresh_name, b"x").unwrap();

        let (names, _) = collect_names(&dir, 1);
        assert_eq!(names, vec!["server-20260101-000000-pid1.log"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn server_logs_ignore_unrelated_files() {
        let dir = server_log_dir("filter");
        fs::write(dir.join("server-20260101-000000-pid1.log"), b"x").unwrap();
        fs::write(dir.join("tauri.log"), b"x").unwrap();
        fs::write(dir.join("server-notes.txt"), b"x").unwrap();

        let (names, _) = collect_names(&dir, 5);
        assert_eq!(names, vec!["server-20260101-000000-pid1.log"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// An entry that matches the name but cannot be read must not spend one of the
    /// two slots: rejecting it later in `read_tail` still costs the real crash log.
    #[test]
    fn unreadable_entries_never_take_a_slot_from_a_real_log() {
        let dir = server_log_dir("crowding");
        let real = dir.join("server-20260101-000000-pid1.log");
        fs::write(&real, b"Current thread 0x1 (most recent call first):\n").unwrap();
        fs::create_dir_all(dir.join("server-20260102-000000-pid2.log")).unwrap();
        #[cfg(unix)]
        {
            let decoy = dir.join("decoy.txt");
            fs::write(&decoy, b"not a session log\n").unwrap();
            std::os::unix::fs::symlink(&decoy, dir.join("server-20260103-000000-pid3.log"))
                .unwrap();
        }

        let (names, warnings) = collect_names(&dir, 2);
        assert_eq!(names, vec!["server-20260101-000000-pid1.log"]);
        assert!(warnings.iter().any(|w| w.contains("non-regular")));
        #[cfg(unix)]
        assert!(warnings.iter().any(|w| w.contains("symlink")));
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A regular file we cannot open is the same defect wearing different clothes:
    /// `stat` needs no permission on the file, so it survives every check above and
    /// spends a slot. A backend once started under sudo leaves two of these behind.
    #[cfg(unix)]
    #[test]
    fn an_unreadable_log_never_takes_a_slot_from_a_readable_one() {
        use std::os::unix::fs::PermissionsExt;
        let dir = server_log_dir("unreadable");
        fs::write(
            dir.join("server-20260101-000000-pid1.log"),
            b"Current thread 0x1 (most recent call first):\n",
        )
        .unwrap();
        let locked: Vec<PathBuf> = [
            "server-20260102-000000-pid2.log",
            "server-20260103-000000-pid3.log",
        ]
        .iter()
        .map(|name| {
            let path = dir.join(name);
            fs::write(&path, b"locked\n").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
            path
        })
        .collect();

        // Mode 000 does not deny root, or anyone holding CAP_DAC_OVERRIDE, so in a
        // root container the fixture cannot express its own precondition and the
        // assertions below would report an environment, not a defect. Probe rather
        // than test the uid, since the capability can be held without being root.
        // Same reason test_recommended_folders_permission.py skips itself as root.
        if File::open(&locked[0]).is_ok() {
            for path in &locked {
                fs::set_permissions(path, fs::Permissions::from_mode(0o644)).unwrap();
            }
            fs::remove_dir_all(&dir).unwrap();
            eprintln!("skipped: this user can read a mode-000 file");
            return;
        }

        let (names, warnings) = collect_names(&dir, 2);
        // Restore before asserting, so a failure still leaves a removable directory.
        for path in &locked {
            fs::set_permissions(path, fs::Permissions::from_mode(0o644)).unwrap();
        }
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(names, vec!["server-20260101-000000-pid1.log"]);
        assert_eq!(
            warnings.iter().filter(|w| w.contains("unreadable")).count(),
            2
        );
    }

    /// A newline in the name forges `key=value` lines in a report a maintainer then
    /// reads as structure.
    #[cfg(unix)]
    #[test]
    fn server_logs_reject_names_that_could_forge_report_structure() {
        let dir = server_log_dir("inject");
        fs::write(dir.join("server-20260101\ninjected=yes\n-pid1.log"), b"x").unwrap();
        fs::write(dir.join("server-20260102-000000-pid2.log"), b"x").unwrap();

        let (names, warnings) = collect_names(&dir, 5);
        assert_eq!(names, vec!["server-20260102-000000-pid2.log"]);
        assert!(warnings.iter().any(|w| w.contains("unprintable name")));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn server_log_section_stays_inside_its_smaller_budget() {
        let dir = server_log_dir("budget");
        let path = dir.join("server-20260101-000000-pid1.log");
        // Wide lines so the BYTE cap binds first: with short ones the line cap alone
        // keeps the section small and raising the byte cap would go unnoticed.
        let wide = "x".repeat(1024);
        let bulk: String = (0..1_000).map(|i| format!("line {i} {wide}\n")).collect();
        fs::write(
            &path,
            format!("{bulk}Current thread 0x1 (most recent call first):\n"),
        )
        .unwrap();

        let mut out = String::new();
        let mut warnings = Vec::new();
        append_tail_section_capped(
            &mut out,
            &path,
            "server.log",
            "backend-session-log",
            SERVER_LOG_TAIL_MAX_LINES,
            SERVER_LOG_TAIL_MAX_BYTES,
            &mut warnings,
        );

        // A literal, not a multiple of the constant under test, which would scale with
        // any raise and never catch one. 80 KiB clears the 64 KiB cap and its header
        // while still catching a raise far short of TAIL_MAX_BYTES; at the 128 KiB
        // this once allowed, a mutation to 100 KiB passed unnoticed.
        assert!(
            out.len() < 80 * 1024,
            "section grew into the phase-log budget: {}",
            out.len()
        );
        assert!(
            TAIL_MAX_BYTES >= 80 * 1024,
            "budget headroom assumption broke"
        );
        // The faulthandler stack sits at the end of the file, so the tail must keep it.
        assert!(out.contains("Current thread 0x1 (most recent call first):"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn server_logs_absent_directory_is_empty() {
        let missing =
            std::env::temp_dir().join(format!("unsloth-no-server-logs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&missing);
        let mut warnings = Vec::new();
        assert!(newest_server_logs_in(&missing, 2, &mut warnings).is_empty());
        // A backend that never started is the ordinary case, not a warning.
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn tail_handles_missing_and_non_utf8() {
        let missing =
            std::env::temp_dir().join(format!("unsloth-missing-tail-{}", std::process::id()));
        assert!(read_tail(&missing, 10, 100).is_err());

        let path =
            std::env::temp_dir().join(format!("unsloth-nonutf8-tail-{}", std::process::id()));
        fs::write(&path, [0xff, b'a', b'\n', b'b']).unwrap();
        let tail = read_tail(&path, 10, 100).unwrap();
        assert!(tail.text.contains('�'));
        let _ = fs::remove_file(path);
    }

    /// The worst of the three. `read_tail` runs on the blocking pool behind the
    /// support-report command, so reaching a FIFO does not skip a log, it wedges the
    /// thread and the report never returns. On a deadline, because a regression here
    /// hangs the suite rather than failing it.
    #[cfg(unix)]
    #[test]
    fn a_fifo_named_like_a_log_cannot_wedge_the_report() {
        use std::ffi::CString;
        let dir = server_log_dir("fifo");
        fs::write(
            dir.join("server-20260101-000000-pid1.log"),
            b"Current thread 0x1 (most recent call first):\n",
        )
        .unwrap();
        let fifo = dir.join("server-20260103-000000-pid3.log");
        let name = CString::new(fifo.as_os_str().to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);

        let (tx, rx) = std::sync::mpsc::channel();
        let probe = dir.clone();
        std::thread::spawn(move || {
            let mut warnings = Vec::new();
            let found = newest_server_logs_in(&probe, 2, &mut warnings);
            let mut out = String::new();
            for path in &found {
                append_tail_section_capped(
                    &mut out,
                    path,
                    "server.log",
                    "backend-session-log",
                    SERVER_LOG_TAIL_MAX_LINES,
                    SERVER_LOG_TAIL_MAX_BYTES,
                    &mut warnings,
                );
            }
            let _ = tx.send(out);
        });
        let out = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("collection blocked on the fifo");
        assert!(out.contains("Current thread 0x1 (most recent call first):"));

        let _ = fs::remove_file(&fifo);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The section sits above the phase logs and enforce_report_limit chops the END,
    /// so a report that overflows the clipboard budget must still carry the stack.
    #[test]
    fn a_truncated_report_still_carries_the_crash_stack() {
        let stack = "Current thread 0x1 (most recent call first):";
        let mut body = String::from("Unsloth Support Diagnostics\n\n== Log tails ==\n");
        body.push_str(&format!(
            "file=server.log source=backend-session-log\n```text\n{stack}\n```\n"
        ));
        body.push_str(&"phase log filler line\n".repeat(60_000));
        let footer = render_footer(&["w".to_string()], &RedactionReport::default());
        assert!(
            body.len() + footer.len() > REPORT_MAX_BYTES,
            "fixture no longer overflows"
        );

        let out = enforce_report_limit(body, footer);
        assert!(out.len() <= REPORT_MAX_BYTES);
        assert!(out.contains("report truncated"));
        assert!(out.contains(stack));
    }
}
