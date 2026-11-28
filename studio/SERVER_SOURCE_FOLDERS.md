# Server source folders

For a remote Studio or Docker installation, the server operator can expose a
specific set of document folders to the owner's browser session. Set
`UNSLOTH_RAG_SOURCE_FOLDERS` to a JSON array of absolute paths before starting
Studio:

```sh
export UNSLOTH_RAG_SOURCE_FOLDERS='["/srv/documents/manuals", "/srv/documents/research"]'
```

For Docker, mount the folders and use their paths **inside the container**:

```yaml
environment:
  UNSLOTH_RAG_SOURCE_FOLDERS: '["/documents/manuals"]'
volumes:
  - ./manuals:/documents/manuals:ro
```

In the browser, open a project's Sources panel or a knowledge base and select
**Add folder**. Choose one of the configured folders. Its documents use the
existing linked-folder indexing and sync behavior; no browser upload is needed.
The desktop app continues to use its native folder picker.

The setting is disabled by default. Only an authenticated owner UI session can
list or link these folders; API keys, keyless callers, and managed accounts
cannot. Requests select an opaque folder ID, never an arbitrary server path.
The normal source-folder restrictions still apply, including refusal of system,
credential, home, filesystem-root, symlink, and managed upload folders. Missing
or invalid folders are omitted from the picker and checked again when linked.

Removing a path from the setting prevents new links after restart. It does not
revoke an existing link; unlink that folder in Studio to stop its sync.
