//! Shared loopback HTTP test doubles for desktop preflight tests.
//!
//! Every server records the request line of each accepted connection so tests
//! can assert not just the classification result but which probes the app
//! actually sent (e.g. that an unverified responder never receives the
//! desktop-login probe).

use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Request line (e.g. `"GET /api/health HTTP/1.1"`) of every accepted connection.
pub(crate) type RequestLog = Arc<Mutex<Vec<String>>>;

/// A canned-response loopback backend: `GET /api/health` answers `health_body`,
/// `POST /api/auth/desktop-login` answers `login_status`, anything else 404s.
pub(crate) struct LoopbackTestServer {
    pub port: u16,
    pub requests: RequestLog,
}

impl LoopbackTestServer {
    /// Bind an ephemeral 127.0.0.1 port.
    pub(crate) async fn bind(health_body: String, login_status: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        Self::serve(listener, health_body, login_status)
    }

    /// Bind the first free desktop candidate port: `probe_existing_backends`
    /// only scans the candidate range, so mutation-guard tests need a port in it.
    pub(crate) async fn bind_candidate_port(
        health_body: String,
        login_status: &'static str,
    ) -> Self {
        let mut listener = None;
        for port in crate::desktop_backend_owner::desktop_candidate_ports() {
            if let Ok(bound) = TcpListener::bind(("127.0.0.1", port)).await {
                listener = Some(bound);
                break;
            }
        }
        let listener = listener.expect("test needs a free desktop preflight port");
        Self::serve(listener, health_body, login_status)
    }

    fn serve(listener: TcpListener, health_body: String, login_status: &'static str) -> Self {
        let port = listener.local_addr().unwrap().port();
        let requests: RequestLog = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut buffer = [0; 2048];
                let Ok(count) = stream.read(&mut buffer).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..count]);
                log.lock()
                    .unwrap()
                    .push(request.lines().next().unwrap_or("").to_string());
                let (status, body) = if request.starts_with("GET /api/health ") {
                    ("200 OK", health_body.as_str())
                } else if request.starts_with("POST /api/auth/desktop-login ") {
                    (login_status, "")
                } else {
                    ("404 Not Found", "")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                if stream.write_all(response.as_bytes()).await.is_err() {
                    break;
                }
            }
        });
        Self { port, requests }
    }
}
