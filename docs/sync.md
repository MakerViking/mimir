# Centralized sync (optional)

Mimir is local-first: every install owns its data in one SQLite file, and by
default **nothing leaves your machine**. This optional layer lets several
installs (laptop, workstation, home server) share their **memories** through a
place you control. It's **off by default** — if you don't configure it, none of
this code runs and the zero-telemetry promise is unchanged.

## What syncs

- **Global memories** (the ones you store with `-g` / `global: true`) and the
  links between them.
- **Project-scoped memories of opted-in projects** — see
  [Project-scoped memories](#project-scoped-memories) below.
- Each install stays authoritative over its own SQLite store; sync is
  uid-keyed **last-write-wins** (newest edit wins, deletes propagate), so it's
  convergent and safe to run from any machine in any order.

**Not synced** (by design): project memories of projects you haven't opted in,
the code graph and indexed docs (they're tied to a specific checkout/folder on
each machine), and your usage signals (strength, recall history). Embeddings
aren't shipped either — each machine recomputes them locally, so there's nothing
heavy on the wire.

## Project-scoped memories

By default only **global** memories sync, because a project's identity is its
absolute path — which differs on every machine (`~/dev/app` here, `~/code/app`
there). To sync a project's memories, give it a **portable key** that's the same
on every checkout, and opt it in:

```sh
cd your-project
mimir project init --sync     # writes .mimir with a stable id + sync = true
git add .mimir && git commit  # so every clone shares the identity
```

`.mimir` is a tiny committed marker:

```toml
id   = "01JZ8X…"   # stable, machine-independent project id (the portable key)
sync = true        # opt this project's memories into sync
```

The portable key is the committed `.mimir` `id` if present, else the normalized
`origin` git remote (e.g. `git:github.com/you/app`), else nothing — in which
case the project stays local. On another machine, pulled project memories attach
to a **shadow** project keyed the same way; the first time you open that project
locally, Mimir adopts the shadow onto your local path. Only memories travel — the
code graph, indexed docs, and embeddings stay per-checkout.

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
always on — a NAS (Unraid/Synology/TrueNAS/QNAP), a Raspberry Pi, a VPS, or any
Docker host.

> **Where *not* to host it:** inside **WSL2** — its NAT'd network isn't reachable
> by other machines without `netsh portproxy` or mirrored-networking mode (WSL2
> is fine as a *client*, though). On ephemeral cloud-container platforms
> (Fly.io/Railway/Render), attach a **persistent volume** — the SQLite store
> must survive restarts.

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

If you don't set `MIMIR_SYNC_TOKEN`, the hub generates one, prints it to its log
(stderr → `docker logs mimir-hub`), and persists it (restarts reuse it).
Retrieve it anytime with **`mimir sync token`** — e.g.
`docker exec mimir-hub mimir sync token` prints just the token on stdout.

**Point each client** at the hub — in `config.toml`:

```toml
[sync]
mode     = "server"
endpoint = "http://your-host:7777"
```

and put the **same token in the environment** (never in the config file, which
is plaintext) — in **two** places:

```sh
# 1. your shell profile, for manual `mimir sync`:
export MIMIR_SYNC_TOKEN=<token>          # ~/.zshenv, ~/.bashrc, fish conf.d, …
```

```jsonc
// 2. the MCP server's env, so background auto-sync runs under your agent.
//    Claude Code — ~/.claude.json (or a project .mcp.json):
"mcpServers": {
  "mimir": { "command": "mimir", "args": ["mcp"],
             "env": { "MIMIR_SYNC_TOKEN": "<token>" } }
}
```

Grab `<token>` from the hub with `mimir sync token`. Then `mimir sync`.

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
mimir sync token      # print the active token (env, else the hub's persisted one)
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

Synced today: global memories and their links, plus opted-in project memories
(via a portable key). Optional future work: an `all-git` mode that keys every
git project by its remote without a committed `.mimir`, and docs/code-graph sync.
