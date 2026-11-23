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
- **Supersession**: a memory you supersede on one machine is superseded on the
  others (when the hub and both machines run a build that carries it). A pin
  on another machine does not hold it off: the memory ends up superseded and
  pinned everywhere.
- **What a memory says is decided by the latest decision about its text.**
  Writing it, rewording it and forgetting it are decisions about the text;
  pinning, tagging, anchoring or giving it an expiry only annotate it. When
  two machines changed one memory without having synced, the later *text*
  decision wins, on every machine:
  - Forget on one machine, pin (or tag, or anchor) on another that had not
    heard: it stays forgotten, however much later the pin was made.
  - Reword on one machine, pin on another: the new wording is kept.
  - Forget on one, reword later on another: the rewording is kept; that is
    someone deciding the memory should say something.
  - Two annotations on the same text: the later one is kept.
  - Two different changes in the very same second: one of them is kept, the
    same one everywhere.
  A machine or hub on a build older than this one still orders by the last
  write of any kind, so until all of them are upgraded a pin made there can
  still bring a forgotten memory back. One thing is never a write: the decay
  pass archiving a memory locally, so it cannot shield a memory from a
  forget or an edit made elsewhere.

**Not synced** (by design): project memories of projects you haven't opted in,
the code graph and indexed docs (they're tied to a specific checkout/folder on
each machine), and your usage signals (strength, recall history). Embeddings
aren't shipped either — each machine recomputes them locally, so there's nothing
heavy on the wire. **Decay archival** is not synced for the same reason as the
usage signals it is computed from: a memory that sat idle on one machine and
was archived there stays live on the machine that uses it.

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
mimir sync status     # mode, endpoint/dir, pending changes, where the token comes from
mimir sync token      # print the active token (env, else the hub's persisted one)
mimir sync push --full   # send every synced memory again (safe to repeat)
mimir sync pull --full   # ask the hub for everything (safe to repeat)
```

### After upgrading the hub

Two things are fixed on the hub and do not reach back in time by themselves:
a hub on 0.17.1 or older never relayed a memory that was pushed long after
it was written, and it dropped the part of a record that says which memory
replaced it. So upgrade in this order:

1. Upgrade the hub and restart `mimir serve`.
2. Upgrade each machine.
3. On each machine, once: `mimir sync push --full`, then `mimir sync pull --full`.

Both are ordinary merges (each machine keeps the winning version of every
memory), so running them twice, or on a machine that missed nothing, changes
nothing.

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
`MIMIR_SYNC_TOKEN` in the MCP server's environment). It re-reads `[sync]` on
every tick, so turning sync or `auto` on or off takes effect within one
interval, without restarting the server.

### The `/m-sync` slash command

`mimir init` installs a `/m-sync` command for your agent CLIs **when sync is
enabled**. After turning sync on in `config.toml`, re-run `mimir init` to add it.

## Troubleshooting

- **`sync is off`** — set `[sync] mode` to `"file"` or `"server"`.
- **`unknown [sync] mode`** — the value is misspelled; it is treated as off
  until it is one of `off`, `file` or `server` (`mimir doctor` names it too).
- **`401` / token rejected** — the client's `MIMIR_SYNC_TOKEN` doesn't match the
  hub's. Check the env var on both sides.
- **`hub unreachable`** — wrong `endpoint`, hub not running, or not reachable on
  your network (is it on the tailnet?).
- **A memory didn't appear** — global memories sync, and project memories of
  projects you opted in (see above); others stay local. If it was written
  while the hub ran an older build, run `mimir sync pull --full` once.
- **Clocks** — last-write-wins compares timestamps, so keep machines roughly
  NTP-synced (a small grace window absorbs minor skew).

## Scope & roadmap

Synced today: global memories and their links, plus opted-in project memories
(via a portable key). Optional future work: an `all-git` mode that keys every
git project by its remote without a committed `.mimir`, and docs/code-graph sync.
