use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Past this many distinct timeouts, clients are built per call.
const MAX_CACHED_CLIENTS: usize = 16;

/// Redirects refused: this client posts `.desktop_secret`, and reqwest strips headers, not bodies,
/// on cross-host redirects, so a 307 would carry the secret off-host.
/// Shared per timeout: a build loads TLS roots. No idle pool: no connection outlives a restart.
pub(crate) fn client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    static CLIENTS: OnceLock<Mutex<HashMap<Duration, reqwest::Client>>> = OnceLock::new();
    let clients = CLIENTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cached) = clients.lock() {
        if let Some(client) = cached.get(&timeout) {
            return Ok(client.clone());
        }
    }
    let client = build_client(timeout)?;
    if let Ok(mut cached) = clients.lock() {
        if cached.len() < MAX_CACHED_CLIENTS {
            cached.entry(timeout).or_insert_with(|| client.clone());
        }
    }
    Ok(client)
}

fn build_client(timeout: Duration) -> Result<reqwest::Client, reqwest::Error> {
    #[cfg(test)]
    tests::record_build(timeout);
    reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .pool_max_idle_per_host(0)
        .build()
}

/// `read_timeout`, not `timeout`: a long clip outlasts any total deadline. Redirects are refused
/// so a loopback URL cannot be bounced off-host after the check.
pub(crate) fn streaming_client(
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::Command;
    use std::time::Duration;

    const CHILD_ENV: &str = "UNSLOTH_TEST_LOOPBACK_HTTP_CHILD";

    static BUILDS: std::sync::Mutex<Vec<Duration>> = std::sync::Mutex::new(Vec::new());

    pub(super) fn record_build(timeout: Duration) {
        BUILDS.lock().unwrap().push(timeout);
    }

    fn builds_of(timeout: Duration) -> usize {
        BUILDS.lock().unwrap().iter().filter(|built| **built == timeout).count()
    }

    #[tokio::test]
    async fn one_client_per_timeout_still_times_out_and_opens_a_connection_per_request() {
        use std::sync::{Arc, Mutex};

        // Unique timeout, so the build count is this test's own.
        let timeout = Duration::from_millis(1_237);
        for _ in 0..5 {
            super::client(timeout).unwrap();
        }
        assert_eq!(builds_of(timeout), 1, "built once, then shared");

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = Arc::new(Mutex::new(0usize));
        let counter = accepted.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            while let Ok((mut socket, _)) = listener.accept().await {
                *counter.lock().unwrap() += 1;
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = socket.read(&mut buf).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        .await;
                    // Kept open: only a pool could reuse it.
                    tokio::time::sleep(Duration::from_secs(3)).await;
                });
            }
        });
        let client = super::client(timeout).unwrap();
        for _ in 0..3 {
            let body = client
                .get(format!("http://127.0.0.1:{port}/"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert_eq!(body, "ok");
        }
        assert_eq!(*accepted.lock().unwrap(), 3);

        let silent = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let silent_port = silent.local_addr().unwrap().port();
        let _hold = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = silent.accept().await {
                held.push(socket);
            }
        });
        let started = std::time::Instant::now();
        let error = super::client(timeout)
            .unwrap()
            .get(format!("http://127.0.0.1:{silent_port}/"))
            .send()
            .await
            .unwrap_err();
        assert!(error.is_timeout(), "{error:?}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn client_ignores_system_proxy() {
        if std::env::var_os(CHILD_ENV).is_none() {
            let current_thread = std::thread::current();
            let test_name = current_thread.name().unwrap();
            let status = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test_name, "--nocapture"])
                .env(CHILD_ENV, "1")
                .env("HTTP_PROXY", "http://127.0.0.1:1")
                .env("http_proxy", "http://127.0.0.1:1")
                .env_remove("HTTPS_PROXY")
                .env_remove("https_proxy")
                .env_remove("ALL_PROXY")
                .env_remove("all_proxy")
                .env_remove("NO_PROXY")
                .env_remove("no_proxy")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(async {
            super::client(Duration::from_secs(2))
                .unwrap()
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .unwrap()
        });
        assert!(response.status().is_success());
        server.join().unwrap();
    }

    /// A followed 307 would resend the desktop secret to the Location host.
    #[test]
    fn a_redirect_is_returned_not_followed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 2048];
            let _ = stream.read(&mut discard);
            let _ = stream.write_all(
                b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://evil.test/collect\r\n\
                  Content-Length: 0\r\nConnection: close\r\n\r\n",
            );
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let response = runtime.block_on(async {
            super::client(Duration::from_secs(2))
                .unwrap()
                .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
                .json(&serde_json::json!({ "secret": "desktop-not-a-real-secret" }))
                .send()
                .await
                .unwrap()
        });

        assert_eq!(response.status().as_u16(), 307);
        assert_eq!(response.url().port(), Some(port));
        server.join().unwrap();
    }
}
