use log::warn;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use super::state::{
    forget_segment_path, record_segment_path, DiagnosticsSnapshot, DiagnosticsState,
};
use super::{
    logs_dir, now_ms, sanitize_component, valid_utf8_boundary, MAX_PHASE_LINE_BYTES,
    PHASE_LOG_KEEP_GROUPS_PER_KIND, PHASE_LOG_MAX_SEGMENTS_PER_GROUP, PHASE_LOG_SEGMENT_MAX_BYTES,
    SCHEMA_VERSION,
};

pub type PhaseLogHandle = Arc<Mutex<PhaseLogWriter>>;

#[derive(Clone, Debug)]
pub(crate) enum SegmentOwner {
    Install {
        attempt_id: String,
    },
    Update {
        attempt_id: String,
    },
    Repair {
        repair_group_id: String,
        attempt_id: String,
        child_type: String,
    },
    Backend {
        session_id: String,
    },
}

pub struct PhaseLogWriter {
    owner: SegmentOwner,
    flow: String,
    started_at_ms: u64,
    backend_kind: Option<String>,
    log_dir: PathBuf,
    segment_index: usize,
    current_bytes: u64,
    segments: Vec<PathBuf>,
    file: Option<File>,
    disabled: bool,
    disable_reason: Option<String>,
    diagnostics: DiagnosticsState,
    max_bytes: u64,
    max_segments: usize,
}

impl PhaseLogWriter {
    pub(crate) fn new(
        diagnostics: DiagnosticsState,
        owner: SegmentOwner,
        flow: impl Into<String>,
        started_at_ms: u64,
        backend_kind: Option<String>,
    ) -> Self {
        let mut writer = Self {
            owner,
            flow: flow.into(),
            started_at_ms,
            backend_kind,
            log_dir: logs_dir(),
            segment_index: 1,
            current_bytes: 0,
            segments: Vec::new(),
            file: None,
            disabled: false,
            disable_reason: None,
            diagnostics,
            max_bytes: PHASE_LOG_SEGMENT_MAX_BYTES,
            max_segments: PHASE_LOG_MAX_SEGMENTS_PER_GROUP,
        };
        writer.open_segment();
        writer
    }

    #[cfg(test)]
    fn new_for_test(
        diagnostics: DiagnosticsState,
        owner: SegmentOwner,
        flow: impl Into<String>,
        started_at_ms: u64,
        log_dir: PathBuf,
        max_bytes: u64,
        max_segments: usize,
    ) -> Self {
        let mut writer = Self {
            owner,
            flow: flow.into(),
            started_at_ms,
            backend_kind: None,
            log_dir,
            segment_index: 1,
            current_bytes: 0,
            segments: Vec::new(),
            file: None,
            disabled: false,
            disable_reason: None,
            diagnostics,
            max_bytes,
            max_segments,
        };
        writer.open_segment();
        writer
    }

    fn open_segment(&mut self) {
        if self.disabled {
            return;
        }
        if let Err(error) = ensure_logs_dir_at(&self.log_dir) {
            self.disable(format!("failed to create diagnostics log dir: {error}"));
            return;
        }
        if self.segment_index == 1 {
            cleanup_retention(&self.log_dir, &self.flow, &owner_group_key(&self.owner));
        }

        let path = self.segment_path(self.segment_index);
        if path_has_symlink(&path) {
            self.disable(format!(
                "refusing to write diagnostics phase log through symlink: {}",
                path.display()
            ));
            return;
        }

        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        match options.open(&path) {
            Ok(mut file) => {
                let header = self.header();
                if let Err(error) = file.write_all(header.as_bytes()) {
                    self.disable(format!(
                        "failed to write diagnostics phase log header to {}: {error}",
                        path.display()
                    ));
                    return;
                }
                self.current_bytes = header.len() as u64;
                self.segments.push(path.clone());
                self.file = Some(file);
                record_segment_path(&self.diagnostics, &self.owner, path);
            }
            Err(error) => self.disable(format!(
                "failed to open diagnostics phase log {}: {error}",
                path.display()
            )),
        }
    }

    fn header(&self) -> String {
        let mut lines = vec![
            "# Unsloth Studio diagnostics phase log".to_string(),
            format!("diag_schema={SCHEMA_VERSION}"),
            format!("flow={}", self.flow),
            format!("segment_index={}", self.segment_index),
            format!("started_at_ms={}", self.started_at_ms),
        ];
        match &self.owner {
            SegmentOwner::Install { attempt_id }
            | SegmentOwner::Update { attempt_id }
            | SegmentOwner::Repair { attempt_id, .. } => {
                lines.push(format!("attempt_id={attempt_id}"));
            }
            SegmentOwner::Backend { session_id } => {
                lines.push(format!("attempt_id={session_id}"));
                lines.push(format!("backend_session_id={session_id}"));
            }
        }
        if let SegmentOwner::Repair {
            repair_group_id,
            child_type,
            ..
        } = &self.owner
        {
            lines.push(format!("repair_group_id={repair_group_id}"));
            lines.push(format!("child_type={child_type}"));
        }
        if let Some(kind) = &self.backend_kind {
            lines.push(format!("backend_kind={kind}"));
        }
        lines.push("--- log ---".to_string());
        lines.push(String::new());
        lines.join("\n")
    }

    fn segment_path(&self, segment_index: usize) -> PathBuf {
        let name = match &self.owner {
            SegmentOwner::Install { attempt_id } => {
                format!(
                    "install-{}-s{segment_index:02}.log",
                    sanitize_component(attempt_id)
                )
            }
            SegmentOwner::Update { attempt_id } => {
                format!(
                    "update-{}-s{segment_index:02}.log",
                    sanitize_component(attempt_id)
                )
            }
            SegmentOwner::Repair {
                repair_group_id,
                attempt_id,
                child_type,
            } => format!(
                "repair-{}-{}-{}-s{segment_index:02}.log",
                sanitize_component(repair_group_id),
                sanitize_component(child_type),
                sanitize_component(attempt_id)
            ),
            SegmentOwner::Backend { session_id } => {
                format!(
                    "backend-{}-s{segment_index:02}.log",
                    sanitize_component(session_id)
                )
            }
        };
        self.log_dir.join(name)
    }

    fn append_line(&mut self, stream: &str, text: &str) {
        if self.disabled {
            return;
        }
        let mut line = text.replace('\r', "");
        let mut truncated = false;
        if line.len() > MAX_PHASE_LINE_BYTES {
            line.truncate(valid_utf8_boundary(&line, MAX_PHASE_LINE_BYTES));
            truncated = true;
        }
        if truncated {
            line.push_str(" [line truncated]");
        }
        let entry = format!("[{}][{}] {}\n", now_ms(), stream, line);
        if self.current_bytes + entry.len() as u64 > self.max_bytes {
            self.rotate();
        }
        let Some(file) = self.file.as_mut() else {
            return;
        };
        match file.write_all(entry.as_bytes()) {
            Ok(()) => {
                self.current_bytes += entry.len() as u64;
            }
            Err(error) => self.disable(format!("diagnostics phase log write failed: {error}")),
        }
    }

    fn rotate(&mut self) {
        self.file.take();
        self.segment_index += 1;
        self.current_bytes = 0;
        self.open_segment();
        self.prune_old_segments();
    }

    fn prune_old_segments(&mut self) {
        while self.segments.len() > self.max_segments {
            let old_path = self.segments.remove(0);
            forget_segment_path(&self.diagnostics, &self.owner, &old_path);
            safe_delete_phase_log(&self.log_dir, &old_path);
        }
    }

    fn disable(&mut self, reason: String) {
        if !self.disabled {
            warn!("{}", reason);
        }
        self.disabled = true;
        self.disable_reason = Some(reason);
        self.file.take();
    }
}

#[allow(dead_code)]
pub fn ensure_logs_dir() -> std::io::Result<PathBuf> {
    let dir = logs_dir();
    ensure_logs_dir_at(&dir)?;
    Ok(dir)
}

pub(crate) fn ensure_logs_dir_at(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    if fs::symlink_metadata(dir)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "refusing diagnostics logs dir symlink",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    }
    Ok(())
}

pub fn append_phase_line(handle: &PhaseLogHandle, stream: &str, text: &str) {
    if let Ok(mut writer) = handle.lock() {
        writer.append_line(stream, text);
    }
}
#[derive(Debug)]
pub(crate) struct PhaseGroup {
    pub(crate) label: String,
    pub(crate) source: String,
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) note: Option<String>,
}

pub(crate) fn collect_phase_groups(
    snapshot: &DiagnosticsSnapshot,
    warnings: &mut Vec<String>,
) -> Vec<PhaseGroup> {
    let mut groups = Vec::new();
    for install in snapshot.install_history.iter().rev().take(4).rev() {
        if !install.log_segments.is_empty() {
            groups.push(PhaseGroup {
                label: format!("install:{}", install.attempt_id),
                source: "live-state-history".to_string(),
                paths: install.log_segments.clone(),
                note: Some("previous linked install attempt retained in memory".to_string()),
            });
        }
    }
    if let Some(install) = &snapshot.install {
        if !install.log_segments.is_empty() {
            groups.push(PhaseGroup {
                label: format!("install:{}", install.attempt_id),
                source: "live-state".to_string(),
                paths: install.log_segments.clone(),
                note: None,
            });
        }
    }
    if let Some(update) = &snapshot.update {
        if !update.log_segments.is_empty() {
            groups.push(PhaseGroup {
                label: format!("update:{}", update.attempt_id),
                source: "live-state".to_string(),
                paths: update.log_segments.clone(),
                note: None,
            });
        }
    }
    for group in snapshot.repair_groups.iter().rev().take(5).rev() {
        let mut paths = Vec::new();
        for child in &group.children {
            paths.extend(child.log_segments.clone());
        }
        if !paths.is_empty() {
            groups.push(PhaseGroup {
                label: format!("repair:{}", group.repair_group_id),
                source: "live-state".to_string(),
                paths,
                note: None,
            });
        }
    }
    if let Some(backend) = &snapshot.backend {
        if backend.backend_kind == "managed" && !backend.log_segments.is_empty() {
            groups.push(PhaseGroup {
                label: backend
                    .session_id
                    .as_ref()
                    .map(|id| format!("backend:{id}"))
                    .unwrap_or_else(|| "backend:unknown".to_string()),
                source: "live-state".to_string(),
                paths: backend.log_segments.clone(),
                note: None,
            });
        } else if backend.backend_kind == "attached_external" {
            groups.push(PhaseGroup {
                label: "backend:attached_external".to_string(),
                source: "live-state".to_string(),
                paths: Vec::new(),
                note: Some("Tauri does not own attached/external backend process logs".to_string()),
            });
        }
    }

    let has_install = groups.iter().any(|g| g.label.starts_with("install:"));
    let has_update = groups.iter().any(|g| g.label.starts_with("update:"));
    let has_repair = groups.iter().any(|g| g.label.starts_with("repair:"));
    let has_backend = groups.iter().any(|g| g.label.starts_with("backend:"));
    let fallback = scan_phase_logs(warnings);
    for flow in ["install", "update", "repair", "backend"] {
        let already_live = match flow {
            "install" => has_install,
            "update" => has_update,
            "repair" => has_repair,
            "backend" => has_backend,
            _ => false,
        };
        if already_live {
            continue;
        }
        if let Some(group) = latest_disk_group(&fallback, flow) {
            groups.push(group);
        }
    }

    groups
}

#[derive(Clone, Debug)]
struct PhaseLogMeta {
    path: PathBuf,
    flow: String,
    group_key: String,
    segment_index: usize,
    started_at_ms: u64,
    mtime_ms: u64,
}

fn scan_phase_logs(warnings: &mut Vec<String>) -> Vec<PhaseLogMeta> {
    scan_phase_logs_in_dir(&logs_dir(), warnings)
}

fn scan_phase_logs_in_dir(dir: &Path, warnings: &mut Vec<String>) -> Vec<PhaseLogMeta> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            warnings.push(format!("phase log dir unavailable: {error}"));
            return Vec::new();
        }
    };
    let mut metas = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_phase_log_name(file_name) {
            continue;
        }
        if path_has_symlink(&path) {
            warnings.push(format!("ignored phase log symlink: {}", path.display()));
            continue;
        }
        match parse_phase_log_meta(&path, file_name) {
            Some(meta) => metas.push(meta),
            None => warnings.push(format!(
                "could not reconstruct phase log metadata: {}",
                path.display()
            )),
        }
    }
    metas
}

fn safe_delete_phase_log(log_dir: &Path, path: &Path) {
    if path_has_symlink(path) {
        return;
    }
    let Ok(canonical_dir) = log_dir.canonicalize() else {
        return;
    };
    let Ok(parent) = path.parent().unwrap_or(log_dir).canonicalize() else {
        return;
    };
    if parent != canonical_dir {
        return;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    if !is_phase_log_name(name) {
        return;
    }
    let _ = fs::remove_file(path);
}

fn cleanup_retention(dir: &Path, flow: &str, current_group_key: &str) {
    let canonical_dir = match dir.canonicalize() {
        Ok(path) => path,
        Err(_) => return,
    };
    let mut warnings = Vec::new();
    let metas = scan_phase_logs_in_dir(dir, &mut warnings);
    let mut groups: BTreeMap<String, Vec<PhaseLogMeta>> = BTreeMap::new();
    for meta in metas.into_iter().filter(|meta| meta.flow == flow) {
        groups.entry(meta.group_key.clone()).or_default().push(meta);
    }
    let mut ordered: Vec<(String, u64)> = groups
        .iter()
        .map(|(key, items)| {
            let newest = items
                .iter()
                .map(|meta| meta.started_at_ms.max(meta.mtime_ms))
                .max()
                .unwrap_or(0);
            (key.clone(), newest)
        })
        .collect();
    ordered.sort_by_key(|(_, newest)| *newest);
    let removable = ordered.len().saturating_sub(PHASE_LOG_KEEP_GROUPS_PER_KIND);
    for (group_key, _) in ordered.into_iter().take(removable) {
        if group_key == current_group_key {
            continue;
        }
        let Some(items) = groups.get(&group_key) else {
            continue;
        };
        for meta in items {
            let Ok(parent) = meta.path.parent().unwrap_or(dir).canonicalize() else {
                continue;
            };
            if parent != canonical_dir {
                continue;
            }
            safe_delete_phase_log(dir, &meta.path);
        }
    }
}

fn owner_group_key(owner: &SegmentOwner) -> String {
    match owner {
        SegmentOwner::Install { attempt_id } => format!("install:{attempt_id}"),
        SegmentOwner::Update { attempt_id } => format!("update:{attempt_id}"),
        SegmentOwner::Repair {
            repair_group_id, ..
        } => format!("repair:{repair_group_id}"),
        SegmentOwner::Backend { session_id } => format!("backend:{session_id}"),
    }
}

fn parse_phase_log_meta(path: &Path, file_name: &str) -> Option<PhaseLogMeta> {
    let mut file = File::open(path).ok()?;
    let mut buf = vec![0u8; 8192];
    let read = file.read(&mut buf).ok()?;
    buf.truncate(read);
    let header = String::from_utf8_lossy(&buf);
    let mut map = HashMap::new();
    for line in header.lines().take(32) {
        if line == "--- log ---" {
            break;
        }
        if let Some((key, value)) = line.split_once('=') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }

    let flow = map
        .get("flow")
        .cloned()
        .or_else(|| file_name.split('-').next().map(str::to_string))?;
    let segment_index = map
        .get("segment_index")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| segment_from_filename(file_name))
        .unwrap_or(1);
    let started_at_ms = map
        .get("started_at_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| file_mtime_ms(path).unwrap_or(0));
    let mtime_ms = file_mtime_ms(path).unwrap_or(started_at_ms);
    let group_key = if flow == "repair" {
        map.get("repair_group_id")
            .cloned()
            .or_else(|| repair_group_from_filename(file_name))
            .map(|id| format!("repair:{id}"))?
    } else if flow == "backend" {
        map.get("backend_session_id")
            .or_else(|| map.get("attempt_id"))
            .cloned()
            .or_else(|| attempt_from_filename(file_name, "backend"))
            .map(|id| format!("backend:{id}"))?
    } else {
        map.get("attempt_id")
            .cloned()
            .or_else(|| attempt_from_filename(file_name, &flow))
            .map(|id| format!("{flow}:{id}"))?
    };

    Some(PhaseLogMeta {
        path: path.to_path_buf(),
        flow,
        group_key,
        segment_index,
        started_at_ms,
        mtime_ms,
    })
}

fn latest_disk_group(metas: &[PhaseLogMeta], flow: &str) -> Option<PhaseGroup> {
    let mut groups: BTreeMap<String, Vec<PhaseLogMeta>> = BTreeMap::new();
    for meta in metas.iter().filter(|m| m.flow == flow) {
        groups
            .entry(meta.group_key.clone())
            .or_default()
            .push(meta.clone());
    }
    let (_, mut selected) = groups
        .into_iter()
        .max_by_key(|(_, items)| items.iter().map(|m| (m.started_at_ms, m.mtime_ms)).max())?;
    selected.sort_by_key(|m| (m.started_at_ms, m.segment_index, m.path.clone()));
    let label = selected
        .first()
        .map(|m| m.group_key.clone())
        .unwrap_or_else(|| format!("{flow}:unknown"));
    Some(PhaseGroup {
        label,
        source: "disk-fallback".to_string(),
        paths: selected.into_iter().map(|m| m.path).collect(),
        note: Some("reconstructed from phase-log headers/filenames".to_string()),
    })
}

pub(crate) fn path_has_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
}

fn file_mtime_ms(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

fn is_phase_log_name(name: &str) -> bool {
    name.ends_with(".log")
        && (name.starts_with("install-")
            || name.starts_with("update-")
            || name.starts_with("repair-")
            || name.starts_with("backend-"))
}

fn segment_from_filename(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".log")?;
    let segment = stem.rsplit_once("-s")?.1;
    segment.parse::<usize>().ok()
}

fn attempt_from_filename(name: &str, flow: &str) -> Option<String> {
    let stem = name.strip_suffix(".log")?;
    let rest = stem.strip_prefix(&format!("{flow}-"))?;
    let (attempt, _) = rest.rsplit_once("-s")?;
    Some(attempt.to_string())
}

fn repair_group_from_filename(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".log")?;
    let rest = stem.strip_prefix("repair-")?;
    let (before_segment, _) = rest.rsplit_once("-s")?;

    // New repair filenames include both child type and attempt id:
    // repair-<repair_group_id>-<child_type>-<attempt_id>-sNN.log.
    // Repair/attempt ids contain hyphens, so split on the deterministic
    // child/attempt prefix pair emitted by SegmentOwner::Repair.
    for child_type in [
        "update",
        "install",
        "elevation",
        "resume",
        "resume_update",
        "resume_install",
    ] {
        let marker = format!("-{child_type}-{child_type}-");
        if let Some((group, _)) = before_segment.split_once(&marker) {
            return Some(group.to_string());
        }
    }

    // Backward-compatible fallback for the older repair-<group>-<child>-sNN form.
    let (group, _) = before_segment.rsplit_once('-')?;
    Some(group.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::state::{clone_snapshot, with_snapshot, AttemptSummary};
    use crate::diagnostics::{new_diagnostics_state, now_ms};

    #[test]
    fn phase_writer_keeps_latest_segments_and_terminal_lines() {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-phase-writer-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let diagnostics = new_diagnostics_state();
        let attempt_id = "install-test".to_string();
        with_snapshot(&diagnostics, |snapshot| {
            snapshot.install = Some(AttemptSummary::new(
                "install",
                attempt_id.clone(),
                None,
                None,
                now_ms(),
            ));
        });
        let mut writer = PhaseLogWriter::new_for_test(
            diagnostics.clone(),
            SegmentOwner::Install {
                attempt_id: attempt_id.clone(),
            },
            "install",
            now_ms(),
            dir.clone(),
            128,
            2,
        );
        for i in 0..100 {
            writer.append_line("stdout", &format!("chunk {i}"));
        }
        writer.append_line("stdout", "terminal failure survives");
        let snapshot = clone_snapshot(&diagnostics);
        let segments = snapshot.install.unwrap().log_segments;
        assert_eq!(segments.len(), 2);
        let total_size: u64 = segments
            .iter()
            .map(|path| fs::metadata(path).unwrap().len())
            .sum();
        assert!(total_size < 1024);
        assert!(segments.iter().any(|path| {
            fs::read_to_string(path)
                .map(|text| text.contains("terminal failure survives"))
                .unwrap_or(false)
        }));
        assert!(!dir.join("install-install-test-s01.log").exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn repair_group_filename_fallback_handles_attempt_id_suffix() {
        let filename = "repair-repair-123-4-install-install-567-8-s01.log";
        assert_eq!(
            repair_group_from_filename(filename),
            Some("repair-123-4".to_string())
        );
    }

    #[test]
    fn repair_children_with_same_type_get_unique_segment_paths() {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-repair-paths-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let diagnostics = new_diagnostics_state();
        let first = PhaseLogWriter::new_for_test(
            diagnostics.clone(),
            SegmentOwner::Repair {
                repair_group_id: "repair-group".to_string(),
                attempt_id: "attempt-one".to_string(),
                child_type: "install".to_string(),
            },
            "repair",
            now_ms(),
            dir.clone(),
            1024,
            2,
        );
        let second = PhaseLogWriter::new_for_test(
            diagnostics,
            SegmentOwner::Repair {
                repair_group_id: "repair-group".to_string(),
                attempt_id: "attempt-two".to_string(),
                child_type: "install".to_string(),
            },
            "repair",
            now_ms(),
            dir.clone(),
            1024,
            2,
        );

        assert_ne!(first.segment_path(1), second.segment_path(1));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_latest_disk_group_by_metadata_not_mtime_only() {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-phase-parse-{}-{}",
            std::process::id(),
            now_ms()
        ));
        fs::create_dir_all(&dir).unwrap();
        let old = dir.join("install-old-s01.log");
        let new = dir.join("install-new-s01.log");
        fs::write(
            &old,
            "diag_schema=1\nflow=install\nattempt_id=old\nsegment_index=1\nstarted_at_ms=1\n--- log ---\nold\n",
        )
        .unwrap();
        fs::write(
            &new,
            "diag_schema=1\nflow=install\nattempt_id=new\nsegment_index=1\nstarted_at_ms=2\n--- log ---\nnew\n",
        )
        .unwrap();
        let metas = vec![
            parse_phase_log_meta(&old, "install-old-s01.log").unwrap(),
            parse_phase_log_meta(&new, "install-new-s01.log").unwrap(),
        ];
        let group = latest_disk_group(&metas, "install").unwrap();
        assert_eq!(group.label, "install:new");
        let _ = fs::remove_dir_all(dir);
    }
}
