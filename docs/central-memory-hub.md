# Running mimir as a central, remotely-accessible memory hub

mimir is local-first, but one store can be shared across machines and reached
from web/mobile AI clients. This guide describes a three-part topology:

1. A central **sync hub** (`mimir serve`) that holds the shared store.
2. Per-machine **mimir clients** that sync to the hub.
3. A **remote MCP endpoint** (`mimir mcp --http`) so browser/mobile AI clients
   (for example claude.ai web and mobile) can reach the same store.

```
  desktop mimir  ──sync──┐
  laptop  mimir  ──sync──┤──>  [ mimir serve ]  (central store, one SQLite DB)
                         │            │
  claude.ai web/mobile ──┘     [ mimir mcp --http ]  (same DB, Streamable-HTTP MCP)
                                      │
                               reverse proxy / tunnel + auth
```

## 1. Central sync hub

Run the hub on an always-on host (NAS, VPS, home server):

```sh
mimir serve --bind 0.0.0.0:<HUB_PORT>
```

The hub just replicates memories; it does no embedding, so it needs no GPU and
very little RAM. A minimal container (prebuilt binary, no Rust build):

```dockerfile
FROM debian:trixie-slim
RUN useradd -m -u 10001 mimir && mkdir -p /data && chown mimir /data
COPY mimir /usr/local/bin/mimir
RUN chmod +x /usr/local/bin/mimir
USER mimir
ENV MIMIR_HOME=/data
EXPOSE <HUB_PORT>
ENTRYPOINT ["mimir", "serve", "--bind", "0.0.0.0:<HUB_PORT>"]
```

Keep `/data` (the store) on persistent, ideally encrypted, storage and back it
up (see below). Protect the hub with a shared sync token.

## 2. Clients sync to the hub

On each machine, point mimir at the hub in `config.toml`:

```toml
[sync]
mode = "server"
endpoint = "http://<HUB_HOST>:<HUB_PORT>"
auto = true            # sync automatically
interval_mins = 30     # how often
```

Sync is last-write-wins. You can also sync on demand:

```sh
mimir sync push     # send local changes to the hub
mimir sync pull     # pull hub changes into the local store
```

The token goes in the environment (for example `MIMIR_SYNC_TOKEN`), never in a
committed file.

## 3. Remote MCP for web/mobile

Browser and mobile AI clients cannot talk to a local stdio MCP server, so serve
MCP over HTTP:

```sh
mimir mcp --http 127.0.0.1:<MCP_PORT>
```

Non-loopback binds are refused unless you add `--http-allow-remote`: the
transport has no auth of its own, so the loopback bind is the security
boundary on a bare host. Inside a container the flag is appropriate — bind
`0.0.0.0` there, but publish the container port to localhost only, as the
compose example below does.

This exposes a Streamable-HTTP MCP server at `/mcp`, plus a plain HTTP
`GET /inject` endpoint on the same bind — **not** an MCP tool, just the warm
counterpart to the opt-in `mimir init --hooks --auto-recall` hook (see the
README's Token savings section). Both endpoints share one process-wide
`Mimir` engine, so `/inject` answers in a few ms once the process is warm,
versus ~280ms for the cold `mimir recall-inject` CLI fallback the hook uses
when no `--http` daemon is reachable. `/inject` carries the *same*
loopback-bind posture as `/mcp` — it's the identical axum server on the
identical address, so the non-loopback refusal and `--http-allow-remote`
escape hatch described above apply to it too.

`GET /inject?prompt=<text>&enrich=<stems>` runs the same relevance-floor
logic as the cold CLI path (`mimir_core::inject::compute`) and returns
`text/plain`: either one formatted `Relevant memory: ...` line, or an empty
200 body when nothing clears the floor — silence is the deliberate default,
not an error. `enrich` is optional, space-separated file-stem hints (the
hook script derives it from `git diff --name-only`); it can extend a real
prompt/memory overlap but can never single-handedly clear the floor. Point
the hook at a remote hub's `/inject` by setting `[hooks] inject_url` in
`config.toml` (or the `MIMIR_INJECT_URL` env var for a one-off override) to
`https://<your-host>/inject` — the same auth/tunnel front-end you put in
front of `/mcp` covers it, since it's the same server.

Point `/mcp` at the **same store** as the hub. Running `mimir serve` and
`mimir mcp --http` against one SQLite DB concurrently is safe (WAL +
busy-timeout). A second container that mounts the hub's data directory works
well:

```yaml
services:
  mimir-mcp:
    build: { context: ., dockerfile: Dockerfile.mcp }
    volumes:
      - <HUB_DATA_DIR>:/data
    # publish only to localhost; the proxy/tunnel reaches it over the network
    ports: ["127.0.0.1:<MCP_PORT>:<MCP_PORT>"]
    entrypoint: ["mimir", "mcp", "--http", "0.0.0.0:<MCP_PORT>", "--http-allow-remote"]
```

## 4. Exposing it safely

The `--http` endpoint has **no authentication of its own** - put an
authenticating reverse proxy or tunnel in front of it (Cloudflare Tunnel +
Access, an OAuth-aware proxy, a VPN, etc.).

Two gotchas:

- **Host-header allowlist (DNS-rebinding protection).** The Streamable-HTTP
  server only accepts `Host: localhost`, `127.0.0.1`, or `::1`. Behind a proxy
  with a public hostname you will get `403 Host header is not allowed`. Fix it
  by making the proxy send `Host: localhost` to the upstream (most proxies and
  Cloudflare Tunnel have a host-header override).
- **Registering the connector.** In the web client, add a custom MCP connector
  pointing at `https://<your-host>/mcp`. The OAuth handshake (OAuth 2.1 + dynamic
  client registration + PKCE) requires the client's auth-callback URLs to be on
  your identity provider's allowed-redirect-URI list.

## 5. Backups

The store is just SQLite. Take periodic logical snapshots and keep them in a
**private** repo:

```sh
mimir export --json > mimir-export.jsonl   # logical, restore-safe
```

Restore into a fresh store with:

```sh
mimir import mimir-export.jsonl
```

A scheduled job (cron, Task Scheduler) that exports and commits the snapshot
daily gives you point-in-time recovery without touching the live DB.
