//! PATH from the login shell for a GUI launch. Adapted from tauri-apps/fix-path-env-rs
//! (MIT OR Apache-2.0) `fix()`, except unsloth#12678: a `-c` shell never loads $HISTFILE
//! but saves history on exit, so the probe disables that save and `exec`s past exit hooks.

const DELIMITER: &str = "_SHELL_ENV_DELIMITER_";

/// Keep the old command; an exception list so a renamed bash/zsh still gets the fix.
const NON_POSIX_SHELLS: &[&str] = &[
    "fish", "nu", "nushell", "csh", "tcsh", "xonsh", "elvish", "pwsh", "ion", "murex",
];

#[cfg_attr(windows, allow(dead_code))]
fn env_command() -> String {
    format!("echo -n \"{DELIMITER}\"; env; echo -n \"{DELIMITER}\"; exit")
}

#[cfg_attr(windows, allow(dead_code))]
fn probe_command(shell: &str) -> String {
    let name = std::path::Path::new(shell)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if NON_POSIX_SHELLS.contains(&name) {
        return env_command();
    }
    // zsh RCS off also covers a readonly HISTFILE. Unset is tried in a subshell first:
    // dash exits on unsetting a readonly variable, even under eval / `|| :`.
    format!(
        "if [ -n \"${{ZSH_VERSION-}}\" ]; then eval 'unsetopt RCS' 2>/dev/null || :; fi; \
         (unset HISTFILE) 2>/dev/null && unset HISTFILE; \
         exec /bin/sh -c 'printf \"%s\" \"{DELIMITER}\"; env; printf \"%s\" \"{DELIMITER}\"'"
    )
}

#[cfg_attr(windows, allow(dead_code))]
fn path_from_output(stdout: &[u8]) -> Option<String> {
    let stdout = String::from_utf8_lossy(stdout);
    let env = stdout.split(DELIMITER).nth(1)?;
    let env = strip_ansi_escapes::strip(env);
    String::from_utf8_lossy(&env)
        .split('\n')
        .filter_map(|line| line.strip_prefix("PATH=").map(str::to_owned))
        .last()
}

#[cfg(not(windows))]
fn read_login_shell_path(shell: &str) -> Option<String> {
    let mut cmd = std::process::Command::new(shell);
    cmd.arg("-ilc")
        .arg(probe_command(shell))
        // Oh My Zsh's auto-update prompt can block the shell forever.
        .env("DISABLE_AUTO_UPDATE", "true");
    if let Some(home) = dirs::home_dir() {
        cmd.current_dir(home);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    path_from_output(&out.stdout)
}

/// Best effort: on any failure PATH stays as the process inherited it.
pub fn fix_path() {
    #[cfg(windows)]
    {
        // std::process::Command can miss a GUI app's PATH until it is set again.
        if let Ok(path) = std::env::var("PATH") {
            std::env::set_var("PATH", path);
        }
    }
    #[cfg(not(windows))]
    {
        let default_shell = if cfg!(target_os = "macos") {
            "/bin/zsh"
        } else {
            "/bin/sh"
        };
        let shell = std::env::var("SHELL").unwrap_or_else(|_| default_shell.into());
        if let Some(path) = read_login_shell_path(&shell) {
            std::env::set_var("PATH", path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_shells_unset_histfile_first() {
        for shell in [
            "/bin/zsh",
            "/bin/bash",
            "/opt/homebrew/bin/bash",
            "/bin/sh",
            "dash",
            "/usr/local/bin/zsh5",
            "/opt/bash-5.2/bin/bash-5.2",
        ] {
            assert!(
                probe_command(shell)
                    .contains("(unset HISTFILE) 2>/dev/null && unset HISTFILE; exec "),
                "{shell}"
            );
        }
        for shell in ["/opt/homebrew/bin/fish", "/usr/bin/nu", "/bin/tcsh"] {
            assert_eq!(probe_command(shell), env_command(), "{shell}");
        }
    }

    #[test]
    fn path_comes_from_between_the_delimiters() {
        let out = format!(
            "motd PATH=/rc/noise\n{DELIMITER}HOME=/h\nPATH=/a:/b=c\n\x1b[0mX=1\n{DELIMITER}"
        );
        assert_eq!(path_from_output(out.as_bytes()).as_deref(), Some("/a:/b=c"));
        assert_eq!(path_from_output(b"no delimiter"), None);
    }

    #[cfg(not(windows))]
    fn run_probe_in(home: &std::path::Path, shell: &str, command: &str) -> Option<String> {
        let out = std::process::Command::new(shell)
            .arg("-ilc")
            .arg(command)
            .env_clear()
            .env("HOME", home)
            .env("ZDOTDIR", home)
            .env("PATH", "/usr/bin:/bin")
            .current_dir(home)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| path_from_output(&out.stdout))
            .flatten()
    }

    /// unsloth#12678 against real shells: the old command loses the history, this one keeps it.
    #[cfg(not(windows))]
    #[test]
    fn reading_path_leaves_the_history_file_alone() {
        const ZSH: &str = "HISTFILE=$HOME/.zsh_history\nHISTSIZE=2000\nSAVEHIST=1000\n\
                           unsetopt APPEND_HISTORY\nprint -s plugin-entry\n";
        const BASH: &str = "HISTSIZE=100000\nHISTFILESIZE=100000\nhistory -s plugin-entry\n";
        // (shell, rc file, history file, rc body, extra file)
        let cases = [
            (
                "bash",
                ".bash_profile",
                ".bash_history",
                BASH.to_string(),
                None,
            ),
            (
                "bash",
                ".bash_profile",
                ".bash_history",
                format!("{BASH}trap 'history -w ~/.bash_history' EXIT\n"),
                None,
            ),
            ("zsh", ".zshrc", ".zsh_history", ZSH.to_string(), None),
            (
                "zsh",
                ".zshrc",
                ".zsh_history",
                ZSH.to_string(),
                Some((".zlogout", "fc -W $HOME/.zsh_history\n")),
            ),
            // Readonly HISTFILE: RCS still stops the save; the failed unset must not abort.
            (
                "zsh",
                ".zshrc",
                ".zsh_history",
                format!("{ZSH}typeset -r HISTFILE SAVEHIST\nsetopt ERR_EXIT\n"),
                None,
            ),
        ];
        let history: String = (0..3000).map(|i| format!("echo history-{i}\n")).collect();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("unsloth-12678-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let mut ran = 0;
        for (index, (shell, rc, hist, body, extra)) in cases.into_iter().enumerate() {
            let Some(shell) = ["/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
                .iter()
                .map(|dir| format!("{dir}/{shell}"))
                .find(|path| std::path::Path::new(path).exists())
            else {
                eprintln!("{shell} not installed, skipped");
                continue;
            };
            ran += 1;
            for (arm, command) in [("before", env_command()), ("after", probe_command(&shell))] {
                let home = root.join(format!("{arm}-{index}"));
                std::fs::create_dir_all(&home).unwrap();
                std::fs::write(home.join(rc), format!("export PATH=/from/rc:$PATH\n{body}"))
                    .unwrap();
                if let Some((name, text)) = extra {
                    std::fs::write(home.join(name), text).unwrap();
                }
                std::fs::write(home.join(hist), &history).unwrap();
                let path = run_probe_in(&home, &shell, &command).unwrap_or_default();
                let kept = std::fs::read_to_string(home.join(hist)).unwrap() == history;
                assert_eq!(
                    kept,
                    arm == "after",
                    "case {index} ({shell}) {arm}: kept = {kept}"
                );
                if arm == "after" {
                    assert!(
                        path.starts_with("/from/rc:"),
                        "case {index} ({shell}): {path:?}"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        assert!(ran > 0, "no POSIX shell found to test against");
    }
}
