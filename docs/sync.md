# Centralized sync (optional)

Mimir is local-first: every install owns its data in one SQLite file, and by
default **nothing leaves your machine**. This optional layer lets several
installs (laptop, workstation, home server) share their **memories** through a
place you control. It's **off by default** — if you don't configure it, none of
this code runs and the zero-telemetry promise is unchanged.

## What syncs

- **Global memories** (the ones you store with `-g` / `global: true`) and the
  links between them.
- Each install stays authoritative over its own SQLite store; sync is
  uid-keyed **last-write-wins** (newest edit wins, deletes propagate), so it's
  convergent and safe to run from any machine in any order.

**Not synced** (by design): project-scoped memories (paths differ per machine —
planned for a later release), the code graph and indexed docs (they're tied to a
specific checkout/folder on each machine), and your usage signals (strength,
recall history). Embeddings aren't shipped either — each machine recomputes them
locally, so there's nothing heavy on the wire.

## Pick a transport

### A. File — the simplest path (no server)

If you already run Syncthing, Dropbox, iCloud Drive, a git repo, or any folder
that syncs between your machines, point Mimir at it. Each machine writes a
snapshot of its memories there and merges everyone else's.

In `config.toml` (`mimir status` shows its path) on **each** machine:

```toml
[sync]
mode = "file"
dir  = "~/Syncthing/mimir"   # a folder your sync tool replicates
```

Then run `mimir sync` (or enable background sync below). That's it — no server,
no ports, no token. Security is whatever your file-sync tool already provides.

### B. Server — a hub you run

For central control, or if you don't use a file-sync tool, run a hub. The same
`mimir` binary is the hub via `mimir serve`; it can live on anything that's
always on — a NAS (Unraid/Synology/TrueNAS), a Raspberry Pi, a VPS, or any
Docker host.

**Run the hub with Docker (recommended):**

```sh
# beside docker-compose.yml, create .env with a token:
echo "MIMIR_SYNC_TOKEN=$(openssl rand -hex 24)" > .env
docker compose up -d        # listens on :7777, data in a named volume
```

**…or run the binary directly** (e.g. under systemd):

```sh
MIMIR_SYNC_TOKEN=$(openssl rand -hex 24) MIMIR_HOME=/srv/mimir mimir serve --bind 0.0.0.0:7777
```

If you don't set `MIMIR_SYNC_TOKEN`, the hub generates one, prints it, and
persists it (restarts reuse it).

**Point each client** at the hub — in `config.toml`:

```toml
[sync]
mode     = "server"
endpoint = "http://your-host:7777"
```

and put the **same token in the environment** (never in the config file, which
is plaintext):

```sh
export MIMIR_SYNC_TOKEN=...   # in your shell profile, and the MCP server's env
```

Then `mimir sync`.

#### Reaching the hub safely

The hub speaks plain HTTP with one bearer token and no rate limiting. **Do not
expose it raw to the public internet.** Two good options:

- **Tailscale / WireGuard (recommended):** bind the hub to your tailnet and use
  the tailnet hostname as the endpoint. Traffic is already encrypted and
  device-authenticated; nothing is publicly reachable.
- **TLS reverse proxy:** put [Caddy](https://caddyserver.com) (automatic HTTPS)
  or nginx in front, and use the `https://` URL as the endpoint.

## Running it

```sh
mimir sync            # full sync (push + pull, or snapshot + merge in file mode)
mimir sync push       # send local changes only
mimir sync pull       # fetch + merge remote changes only
mimir sync status     # mode, endpoint/dir, pending changes, token presence
```

### Background sync

To sync automatically while your agent works, set `auto` and a cadence:

```toml
[sync]
mode = "server"            # or "file"
endpoint = "http://your-host:7777"
auto = true
interval_mins = 30
```

The MCP server then syncs on start and every `interval_mins` (server mode needs
`MIMIR_SYNC_TOKEN` in the MCP server's environment).

### The `/m-sync` slash command

`mimir init` installs a `/m-sync` command for your agent CLIs **when sync is
enabled**. After turning sync on in `config.toml`, re-run `mimir init` to add it.

## Troubleshooting

- **`sync is off`** — set `[sync] mode` to `"file"` or `"server"`.
- **`401` / token rejected** — the client's `MIMIR_SYNC_TOKEN` doesn't match the
  hub's. Check the env var on both sides.
- **`hub unreachable`** — wrong `endpoint`, hub not running, or not reachable on
  your network (is it on the tailnet?).
- **A memory didn't appear** — only **global** memories sync. Store with `-g`
  (CLI) or `global: true` (MCP). Project-scoped memories stay local for now.
- **Clocks** — last-write-wins compares timestamps, so keep machines roughly
  NTP-synced (a small grace window absorbs minor skew).

## Scope & roadmap

This is the MVP: global memories + their links. Planned next: project-scoped
memories (mapped across machines), and optionally docs/code-graph sync.
