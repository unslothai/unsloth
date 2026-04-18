# Graph Report - worked/httpx/raw  (2026-04-05)

## Corpus Check
- 6 files · ~2,047 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 144 nodes · 330 edges · 6 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Client` - 26 edges
2. `AsyncClient` - 25 edges
3. `Response` - 24 edges
4. `Request` - 21 edges
5. `BaseClient` - 18 edges
6. `HTTPTransport` - 17 edges
7. `BaseTransport` - 16 edges
8. `AsyncHTTPTransport` - 15 edges
9. `Headers` - 15 edges
10. `Timeout` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Timeout` --uses--> `URL`  [INFERRED]
  worked/httpx/raw/client.py → worked/httpx/raw/models.py
- `Timeout` --uses--> `Headers`  [INFERRED]
  worked/httpx/raw/client.py → worked/httpx/raw/models.py
- `Timeout` --uses--> `Cookies`  [INFERRED]
  worked/httpx/raw/client.py → worked/httpx/raw/models.py
- `Timeout` --uses--> `BaseTransport`  [INFERRED]
  worked/httpx/raw/client.py → worked/httpx/raw/transport.py
- `Timeout` --uses--> `HTTPTransport`  [INFERRED]
  worked/httpx/raw/client.py → worked/httpx/raw/transport.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (8): ConnectError, AsyncBaseTransport, AsyncHTTPTransport, BaseTransport, ConnectionPool, HTTPTransport, MockTransport, ProxyTransport

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (9): Auth, BasicAuth, BearerAuth, DigestAuth, NetRCAuth, Limits, Timeout, Request (+1 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (3): AsyncClient, BaseClient, Client

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (3): Cookies, Headers, URL

### Community 4 - "Community 4"
Cohesion: 0.16
Nodes (20): Exception, CloseError, ConnectTimeout, CookieConflict, DecodingError, HTTPError, HTTPStatusError, InvalidURL (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.28
Nodes (3): build_url_with_params(), flatten_queryparams(), primitive_value_to_str()

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Client` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.177) - this node is a cross-community bridge._
- **Why does `Response` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `AsyncClient` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `Client` (e.g. with `Request` and `Response`) actually correct?**
  _`Client` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `AsyncClient` (e.g. with `Request` and `Response`) actually correct?**
  _`AsyncClient` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `Response` (e.g. with `Timeout` and `Limits`) actually correct?**
  _`Response` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `Request` (e.g. with `Timeout` and `Limits`) actually correct?**
  _`Request` has 18 INFERRED edges - model-reasoned connections that need verification._