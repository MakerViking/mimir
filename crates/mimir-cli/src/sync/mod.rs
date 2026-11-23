//! `mimir sync` / `mimir serve` — the OPTIONAL centralized-sync layer.
//!
//! Two transports share one replication core (mimir_core::replicate):
//! `file` (a folder replicated by Syncthing/Dropbox/…) and `server` (a
//! `mimir serve` hub).
//!
//! `[sync] mode` is read in exactly one place, [`Mode::of`]. `sync`, `push`
//! and `pull` need a transport and refuse `off`; `status` describes whatever
//! the mode is; all four reject a value that is none of the three. `serve`
//! and `token` do not read the mode at all: a hub is an ordinary store, and
//! usually one with sync off.
//!
//! Transport functions return summaries instead of printing, so both the CLI
//! (which prints) and the MCP background loop (which logs via tracing, since
//! stdout is the MCP protocol channel) can reuse them.

mod file;
mod http;
mod server;

pub(crate) use server::ct_eq;

use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use mimir_core::config::{Config, SyncConfig};
use mimir_core::replicate::{self, ApplyStats};
use mimir_core::Mimir;
use rusqlite::Connection;

/// Watermarks are persisted minus this grace window so a record written during
/// a sync round (clock skew / same-second) re-delivers next time; idempotent
/// last-write-wins apply makes the re-delivery free.
const GRACE_SECS: i64 = 60;

/// Read an env var, treating empty-but-set as unset — an unset shell variable
/// interpolated into a flag or env assignment must not count as configuration.
pub(crate) fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|t| !t.is_empty())
}

/// `[sync] mode`, parsed. The config keeps it as a string, and four functions
/// here used to match on that string with four different answers for a value
/// that was none of the three: an error with a hint, an error without one,
/// and — in `status` — no complaint at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    File,
    Server,
}

impl Mode {
    /// The mode `config` asks for. A value that is not a mode is an error
    /// here, once, so every caller reports it the same way.
    fn of(config: &SyncConfig) -> Result<Mode> {
        match config.mode.as_str() {
            "off" => Ok(Mode::Off),
            "file" => Ok(Mode::File),
            "server" => Ok(Mode::Server),
            other => bail!(
                "unknown [sync] mode \"{other}\" in config.toml — use \"off\", \"file\" or \
                 \"server\""
            ),
        }
    }
}

/// What a command that needs a transport says when there is none.
fn sync_is_off() -> anyhow::Error {
    anyhow::anyhow!(
        "sync is off. Enable it in config.toml under [sync] (mode = \"file\" or \
         \"server\") — see docs/sync.md."
    )
}

/// Cursor advancement for a sync watermark: the batch's high watermark,
/// clamped to the local wall clock (then minus [`GRACE_SECS`], floored at 0).
/// A single future-dated row — our own past clock skew, or a skewed peer's
/// row relayed by the hub — inflates the batch max arbitrarily; unclamped,
/// the cursor would jump past every normally-timestamped later change and
/// this client would silently stop sending (push) or receiving (pull) them.
/// Clamped, the worst case is the skewed row re-transferring each round,
/// which the idempotent apply makes free.
fn cursor_advance(batch_watermark: i64) -> i64 {
    (batch_watermark.min(mimir_core::model::now_unix()) - GRACE_SECS).max(0)
}

/// The watermark a stored cursor stands for: [`cursor_advance`] undone.
///
/// The cursor is deliberately a minute behind what was actually sent, so
/// that minute is sent again next round. It is the right place to *resume*
/// from and the wrong place to *count* from: the newest row pushed is always
/// inside that minute, so counting from the cursor reports it as still
/// pending forever. A cursor of 0 is a store that has not pushed.
fn watermark_behind(cursor: i64) -> i64 {
    if cursor == 0 {
        0
    } else {
        cursor + GRACE_SECS
    }
}

/// Memory changes the hub has not been sent: rows newer than the last pushed
/// watermark (server mode only — nothing keeps a push cursor in file mode,
/// where every sync writes the whole snapshot).
fn pending_push(conn: &Connection) -> Result<usize> {
    let pushed = watermark_behind(replicate::get_watermark(conn, "last_push")?);
    Ok(replicate::changes_since(conn, pushed)?.nodes.len())
}

/// Mode-appropriate full sync, returning a one-line summary (no printing).
/// Used by the CLI `sync` command and the MCP background loop.
pub(crate) fn perform(m: &mut Mimir) -> Result<String> {
    match Mode::of(&m.config.sync)? {
        Mode::Off => Err(sync_is_off()),
        Mode::File => {
            let dir = file_dir(m)?;
            let s = file::sync(m, &dir)?;
            Ok(format!(
                "file: wrote {} memories, merged {} peer file(s) → {} applied, {} deletes, {} edges{}",
                s.wrote,
                s.peers_merged,
                s.applied.nodes_upserted,
                s.applied.tombstones,
                s.applied.edges_upserted,
                skipped_note(s.peers_skipped)
            ))
        }
        Mode::Server => {
            let (sent, pushed) = srv_push(m, false)?;
            let pulled = srv_pull(m, false)?;
            Ok(format!(
                "server: pushed {sent} (hub applied {}), pulled {} applied, {} deletes, {} edges",
                pushed.applied.nodes_upserted,
                pulled.nodes_upserted,
                pulled.tombstones,
                pulled.edges_upserted
            ))
        }
    }
}

/// The tail of a file-mode summary: silent when every peer file was read.
fn skipped_note(skipped: usize) -> String {
    if skipped == 0 {
        String::new()
    } else {
        format!(", {skipped} unreadable peer file(s) skipped")
    }
}

/// Full sync (the default `mimir sync`).
pub fn sync() -> Result<()> {
    let mut m = Mimir::open()?;
    println!("{}", perform(&mut m)?);
    Ok(())
}

/// `full` sends a hub the whole synced set instead of what changed since the
/// last push. File mode writes a whole snapshot every time, so there it
/// changes nothing.
pub fn push(full: bool) -> Result<()> {
    let mut m = Mimir::open()?;
    match Mode::of(&m.config.sync)? {
        Mode::Off => return Err(sync_is_off()),
        Mode::File => {
            let dir = file_dir(&m)?;
            let n = file::write_snapshot(&m, &dir)?;
            println!("wrote snapshot of {n} memories → {}", dir.display());
        }
        Mode::Server => {
            let (sent, res) = srv_push(&mut m, full)?;
            println!(
                "pushed {sent} {} → hub applied {}",
                if full {
                    "memories (everything)"
                } else {
                    "change(s)"
                },
                res.applied.nodes_upserted
            );
        }
    }
    Ok(())
}

/// `full` asks a hub for its whole synced set instead of what arrived since
/// the last pull. File mode merges whole peer snapshots every time, so there
/// it changes nothing.
pub fn pull(full: bool) -> Result<()> {
    let mut m = Mimir::open()?;
    match Mode::of(&m.config.sync)? {
        Mode::Off => return Err(sync_is_off()),
        Mode::File => {
            let dir = file_dir(&m)?;
            let merged = file::merge_peers(&mut m, &dir)?;
            let _ = m.embed_pending();
            println!(
                "merged {} peer file(s) → {} applied, {} deletes, {} edges{}",
                merged.peers,
                merged.applied.nodes_upserted,
                merged.applied.tombstones,
                merged.applied.edges_upserted,
                skipped_note(merged.skipped)
            );
        }
        Mode::Server => {
            let s = srv_pull(&mut m, full)?;
            println!(
                "pulled{} → {} applied, {} deletes, {} edges",
                if full { " everything" } else { "" },
                s.nodes_upserted,
                s.tombstones,
                s.edges_upserted
            );
        }
    }
    Ok(())
}

pub fn status() -> Result<()> {
    let m = Mimir::open()?;
    let sc = &m.config.sync;
    let mode = Mode::of(sc)?;
    println!("mode      {}", sc.mode);
    match mode {
        Mode::Off => {
            println!("(sync is off — see docs/sync.md to enable)");
            return Ok(());
        }
        // File mode keeps no cursor: every sync writes the whole snapshot and
        // merges every peer file, so there is no "pending" to count.
        Mode::File => println!("dir       {}", dir_label(&m)),
        Mode::Server => {
            println!("endpoint  {}", sc.endpoint);
            // The source, never the value: `sync token` is the command that
            // prints it.
            let token = match TokenSource::find(|| stored_token(&m.conn))? {
                TokenSource::Env(_) => "env (MIMIR_SYNC_TOKEN)",
                TokenSource::Stored(_) => {
                    "stored (this store's own hub token; push and pull read MIMIR_SYNC_TOKEN)"
                }
                TokenSource::Missing => "missing — set MIMIR_SYNC_TOKEN",
            };
            println!("token     {token}");
            println!(
                "pending   {} memory change(s) to push",
                pending_push(&m.conn)?
            );
        }
    }
    println!("auto      {} (every {} min)", sc.auto, sc.interval_mins);
    Ok(())
}

/// The sync half of `mimir status` and of the MCP `status` tool, without the
/// `sync    ` label: what the mode is, and the one or two facts that mode
/// has. An endpoint and cursors exist only in server mode; printing them for
/// a file-mode store showed an empty endpoint and two zeros that nothing
/// would ever move. Never fails — a bad mode is reported in the line, since
/// `status` is how one finds out.
pub(crate) fn status_line(m: &Mimir) -> String {
    let sc = &m.config.sync;
    let cadence = if sc.auto {
        format!("auto every {} min", sc.interval_mins)
    } else {
        "manual".into()
    };
    match Mode::of(sc) {
        Ok(Mode::Off) => "off (local store only)".into(),
        Ok(Mode::File) => format!("file {} ({cadence})", dir_label(m)),
        Ok(Mode::Server) => {
            let push = replicate::get_watermark(&m.conn, "last_push").unwrap_or(0);
            let pull = replicate::get_watermark(&m.conn, "last_pull").unwrap_or(0);
            format!(
                "server {} ({cadence}); local watermarks push={push} pull={pull}",
                sc.endpoint
            )
        }
        Err(e) => e.to_string(),
    }
}

/// What the MCP server's background loop does next, decided from the config
/// as it is now rather than as it was when the server started: whether to
/// sync, and how long to wait before asking again.
#[derive(Debug, PartialEq, Eq)]
struct AutoPlan {
    run: bool,
    pause: Duration,
}

impl AutoPlan {
    fn of(config: &SyncConfig) -> AutoPlan {
        AutoPlan {
            // Off is not a failure and has nothing to say. An unknown mode
            // runs, so `perform` reports it: a typo in the config should not
            // be indistinguishable from sync being off.
            run: config.auto && !matches!(Mode::of(config), Ok(Mode::Off)),
            pause: Duration::from_secs(config.interval_mins.max(1) * 60),
        }
    }
}

/// One tick of the MCP server's background sync: re-read `[sync]`, sync if
/// it says to, and return how long to sleep before the next tick.
///
/// The loop used to freeze "enabled", `auto` and the interval at startup
/// while re-reading the mode on each pass, so switching sync off under a
/// running server logged "auto-sync failed: sync is off" every interval
/// until restart, and switching it on did nothing. Only the config file is
/// read when there is nothing to do; `open` (which opens the store) runs
/// only for a sync.
pub(crate) fn auto_tick(config_file: &Path, open: impl FnOnce() -> Result<Mimir>) -> Duration {
    let config = match Config::load(config_file) {
        Ok(config) => config.sync,
        Err(e) => {
            tracing::warn!("auto-sync: {e}");
            return AutoPlan::of(&SyncConfig::default()).pause;
        }
    };
    let plan = AutoPlan::of(&config);
    if plan.run {
        match open() {
            Ok(mut m) => match perform(&mut m) {
                Ok(summary) => tracing::info!("auto-sync: {summary}"),
                Err(e) => tracing::warn!("auto-sync failed: {e}"),
            },
            Err(e) => tracing::warn!("auto-sync open failed: {e}"),
        }
    }
    plan.pause
}

/// Run the `mimir serve` hub.
pub fn serve(bind: String) -> Result<()> {
    server::serve(&bind)
}

/// Where a sync token can come from. `sync token`, `sync status` and the hub
/// at startup all resolve it through [`TokenSource::find`], so they cannot
/// disagree about whether there is one.
enum TokenSource {
    /// `MIMIR_SYNC_TOKEN` in the environment. Wins over a stored one.
    Env(String),
    /// The token a hub generated and persisted in this store's meta table.
    Stored(String),
    Missing,
}

impl TokenSource {
    /// `stored` is asked only when the environment has no token, so a caller
    /// that would have to open the store for it does not when it need not.
    fn find(stored: impl FnOnce() -> Result<Option<String>>) -> Result<TokenSource> {
        if let Some(t) = env_nonempty("MIMIR_SYNC_TOKEN") {
            return Ok(TokenSource::Env(t));
        }
        Ok(match stored()? {
            Some(t) => TokenSource::Stored(t),
            None => TokenSource::Missing,
        })
    }
}

fn stored_token(conn: &Connection) -> Result<Option<String>> {
    Ok(replicate::get_str_meta(conn, "server_token")?)
}

/// Print the active sync token so it can be copied to a client: the
/// `MIMIR_SYNC_TOKEN` env var if set, else the hub's persisted token (server
/// mode). The token goes to stdout (scriptable, e.g. `mimir sync token`); the
/// provenance note goes to stderr.
pub fn token() -> Result<()> {
    match TokenSource::find(|| stored_token(&Mimir::open()?.conn))? {
        TokenSource::Env(t) => {
            println!("{t}");
            eprintln!("(from MIMIR_SYNC_TOKEN in the environment)");
        }
        TokenSource::Stored(t) => {
            println!("{t}");
            eprintln!("(this hub's persisted token — set MIMIR_SYNC_TOKEN to override)");
        }
        TokenSource::Missing => bail!(
            "no sync token found — set MIMIR_SYNC_TOKEN, or start the hub once \
             (`mimir serve`) to generate and persist one"
        ),
    }
    Ok(())
}

// ---- helpers ----

fn file_dir(m: &Mimir) -> Result<std::path::PathBuf> {
    if m.config.sync.dir.is_empty() {
        bail!("file sync needs `[sync] dir = \"...\"` in config.toml (a folder your file-sync tool replicates)");
    }
    Ok(std::path::PathBuf::from(expand_tilde(&m.config.sync.dir)))
}

/// The file-mode directory as a status surface shows it: the path sync will
/// really use, or the reason there is none.
fn dir_label(m: &Mimir) -> String {
    match file_dir(m) {
        Ok(dir) => dir.display().to_string(),
        Err(_) => "(not set — add `[sync] dir = \"...\"` to config.toml)".into(),
    }
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(base) = directories::BaseDirs::new() {
            return base.home_dir().join(rest).to_string_lossy().into_owned();
        }
    }
    p.to_string()
}

fn server_token() -> Result<String> {
    env_nonempty("MIMIR_SYNC_TOKEN")
        .context("server-mode sync needs MIMIR_SYNC_TOKEN in the environment")
}

fn server_endpoint(m: &Mimir) -> Result<String> {
    if m.config.sync.endpoint.is_empty() {
        bail!("server sync needs `[sync] endpoint = \"http://...\"` in config.toml");
    }
    Ok(m.config.sync.endpoint.clone())
}

/// Push local changes; returns (count sent, hub result). Advances last_push.
///
/// `full` sends from the beginning instead of from the stored cursor. A
/// record carries more than it used to (which memory replaced it); a hub on
/// an older build dropped that, and the cursor has long since passed those
/// rows, so nothing would ever send them again. The hub merges by
/// last-write-wins either way, so sending everything changes only what the
/// hub is missing.
fn srv_push(m: &mut Mimir, full: bool) -> Result<(usize, http::PushResult)> {
    let (endpoint, token) = (server_endpoint(m)?, server_token()?);
    let since = if full {
        i64::MIN
    } else {
        replicate::get_watermark(&m.conn, "last_push")?
    };
    let batch = replicate::changes_since(&m.conn, since)?;
    let sent = batch.nodes.len();
    let res = http::push(&endpoint, &token, &batch)?;
    // Advance last_push from OUR batch's high watermark (local clock domain,
    // clamped to the local clock — see `cursor_advance`), never the hub's
    // post-apply global watermark: that one is a cross-peer clock domain a
    // skewed peer can inflate (see e2e `server_sync_survives_peer_clock_skew`).
    // The hub still sends its watermark on the wire; the client-side
    // `PushResult` deliberately has no field for it.
    replicate::set_watermark(&m.conn, "last_push", cursor_advance(batch.watermark))?;
    Ok((sent, res))
}

/// Pull + apply remote changes; returns apply stats. Advances last_pull and
/// embeds new arrivals locally.
/// `full` pulls from the beginning instead of from the stored cursor. The
/// cursor is exact only against a hub that relays by arrival; rows an older
/// hub received late sit below it for good, and a pull from zero is the one
/// way to collect them. It is as safe as any pull: nothing already here is
/// overwritten unless the hub's copy is newer.
fn srv_pull(m: &mut Mimir, full: bool) -> Result<ApplyStats> {
    let (endpoint, token) = (server_endpoint(m)?, server_token()?);
    let since = if full {
        0
    } else {
        replicate::get_watermark(&m.conn, "last_pull")?
    };
    let batch = http::pull(&endpoint, &token, since)?;
    let applied = replicate::apply_changes(&m.conn, &batch)?;
    // The pull cursor lives in the hub's clock: a current hub stamps the
    // batch with its own time (`replicate::relay_since`) and selects by when
    // rows reached it. A hub on an older build sends the newest row timestamp
    // instead, which one skewed row can put years ahead — hence the same
    // clamp as the push cursor. Against a current hub the clamp only bites
    // when the hub's clock runs ahead of ours, and then it errs towards
    // receiving the last stretch again.
    replicate::set_watermark(&m.conn, "last_pull", cursor_advance(batch.watermark))?;
    let _ = m.embed_pending();
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A watermark from the future must not carry the cursor with it. A hub
    /// on an older build answers a pull with its newest row timestamp, and
    /// one row from a machine whose clock ran ahead would put the cursor
    /// past everything written since: this client would stop receiving, and
    /// say nothing. The clamp is all that stands in the way.
    #[test]
    fn a_cursor_never_runs_ahead_of_the_local_clock() {
        let now = mimir_core::model::now_unix();
        let from_the_future = cursor_advance(now + 10 * 365 * 86_400);
        assert!(from_the_future <= now, "{from_the_future} > {now}");
        assert!(from_the_future >= now - GRACE_SECS - 2);
        // An honest watermark is kept, minus the grace window.
        assert_eq!(cursor_advance(now - 1_000), now - 1_000 - GRACE_SECS);
        assert_eq!(cursor_advance(5), 0, "never negative");
    }

    fn config(mode: &str, auto: bool, interval_mins: u64) -> SyncConfig {
        SyncConfig {
            mode: mode.into(),
            auto,
            interval_mins,
            ..SyncConfig::default()
        }
    }

    #[test]
    fn mode_is_one_of_three_or_an_error_that_names_them() {
        assert_eq!(Mode::of(&config("off", false, 30)).unwrap(), Mode::Off);
        assert_eq!(Mode::of(&config("file", false, 30)).unwrap(), Mode::File);
        assert_eq!(
            Mode::of(&config("server", false, 30)).unwrap(),
            Mode::Server
        );
        for typo in ["sever", "Server", "", "on"] {
            let err = Mode::of(&config(typo, false, 30)).unwrap_err().to_string();
            for valid in ["\"off\"", "\"file\"", "\"server\""] {
                assert!(err.contains(valid), "{err}");
            }
            assert!(err.contains(&format!("\"{typo}\"")), "{err}");
        }
    }

    /// `cursor_advance` and `watermark_behind` are each other's inverse for
    /// every watermark a real push can have produced.
    #[test]
    fn a_stored_cursor_reads_back_as_the_watermark_it_was_cut_from() {
        let now = mimir_core::model::now_unix();
        for watermark in [now, now - 5, now - 86_400] {
            assert_eq!(watermark_behind(cursor_advance(watermark)), watermark);
        }
        // Never pushed, and an empty first push.
        assert_eq!(watermark_behind(0), 0);
        assert_eq!(watermark_behind(cursor_advance(0)), 0);
        // A future-dated row: the cursor was clamped to the clock, so the row
        // stays above it — and it is in fact re-sent every round.
        assert!(watermark_behind(cursor_advance(now + 86_400)) <= now + 1);
    }

    fn store(root: &Path) -> Mimir {
        std::fs::create_dir_all(root).unwrap();
        Mimir::open_at(mimir_core::config::Paths::under_root(root)).unwrap()
    }

    fn remember(m: &Mimir, text: &str) {
        mimir_core::memory::remember(
            &m.conn,
            mimir_core::memory::Remember {
                text: text.into(),
                mtype: mimir_core::model::MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: true,
                actor: mimir_core::model::Actor::System,
            },
        )
        .unwrap();
    }

    /// The push cursor is where the next push resumes, a minute behind what
    /// was sent. Counted from there, the newest pushed row is pending
    /// forever; counted from what was sent, it is not.
    #[test]
    fn pending_is_counted_from_what_was_pushed() {
        let tmp = tempfile::tempdir().unwrap();
        let m = store(tmp.path());
        remember(&m, "the first note");
        assert_eq!(pending_push(&m.conn).unwrap(), 1, "never pushed");

        // What `srv_push` records once the hub has answered.
        let batch = replicate::changes_since(&m.conn, 0).unwrap();
        replicate::set_watermark(&m.conn, "last_push", cursor_advance(batch.watermark)).unwrap();
        let cursor = replicate::get_watermark(&m.conn, "last_push").unwrap();
        assert_eq!(
            replicate::changes_since(&m.conn, cursor)
                .unwrap()
                .nodes
                .len(),
            1,
            "the next push does re-send it, inside the grace window"
        );
        assert_eq!(pending_push(&m.conn).unwrap(), 0, "but the hub has it");

        // A change made after the push is pending. Stepped a second ahead
        // rather than slept past the timestamp resolution.
        remember(&m, "the second note");
        m.conn
            .execute(
                "UPDATE node SET updated_at = updated_at + 1 WHERE body = 'the second note'",
                [],
            )
            .unwrap();
        assert_eq!(pending_push(&m.conn).unwrap(), 1);
    }

    #[test]
    fn the_status_line_carries_what_the_mode_has_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();
        let mut m = store(tmp.path());
        assert_eq!(status_line(&m), "off (local store only)");

        m.config.sync.mode = "file".into();
        m.config.sync.dir = "/srv/mimir-sync".into();
        assert_eq!(status_line(&m), "file /srv/mimir-sync (manual)");
        m.config.sync.dir = String::new();
        assert!(
            status_line(&m).starts_with("file (not set"),
            "{}",
            status_line(&m)
        );

        m.config.sync.mode = "server".into();
        m.config.sync.endpoint = "http://hub.tailnet:7777".into();
        m.config.sync.auto = true;
        m.config.sync.interval_mins = 15;
        replicate::set_watermark(&m.conn, "last_push", 100).unwrap();
        replicate::set_watermark(&m.conn, "last_pull", 200).unwrap();
        assert_eq!(
            status_line(&m),
            "server http://hub.tailnet:7777 (auto every 15 min); \
             local watermarks push=100 pull=200"
        );

        m.config.sync.mode = "sever".into();
        assert!(
            status_line(&m).contains("unknown [sync] mode \"sever\""),
            "{}",
            status_line(&m)
        );
    }

    #[test]
    fn the_background_loop_plans_from_the_config_it_is_given() {
        let plan = |mode, auto, mins| AutoPlan::of(&config(mode, auto, mins));
        let mins = |m: u64| Duration::from_secs(m * 60);
        assert_eq!(
            plan("server", true, 30),
            AutoPlan {
                run: true,
                pause: mins(30)
            }
        );
        assert!(plan("file", true, 30).run);
        assert!(!plan("off", true, 30).run, "off is not run, so not logged");
        assert!(!plan("server", false, 30).run, "auto is opt-in");
        assert!(plan("sever", true, 30).run, "a typo is reported, not muted");
        assert_eq!(plan("off", false, 5).pause, mins(5));
        assert_eq!(plan("off", false, 0).pause, mins(1), "never a busy loop");
    }

    /// The loop's whole state is the config file: each tick acts on what the
    /// file says at that moment.
    #[test]
    fn each_tick_rereads_enabled_auto_and_interval() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("config.toml");
        let tick = |toml: &str| {
            std::fs::write(&file, toml).unwrap();
            let mut opened = false;
            let pause = auto_tick(&file, || {
                opened = true;
                bail!("this test has no store to sync")
            });
            (opened, pause.as_secs() / 60)
        };

        let on = "[sync]\nmode = \"server\"\nendpoint = \"http://127.0.0.1:1\"\nauto = true\n";
        assert_eq!(tick(&format!("{on}interval_mins = 5\n")), (true, 5));
        // Switched off under a running server: nothing is opened, so nothing
        // fails and nothing is logged.
        assert_eq!(
            tick("[sync]\nmode = \"off\"\nauto = true\ninterval_mins = 5\n"),
            (false, 5)
        );
        assert_eq!(
            tick("[sync]\nmode = \"server\"\nauto = false\ninterval_mins = 7\n"),
            (false, 7)
        );
        // And back on, with a new interval, without a restart.
        assert_eq!(tick(&format!("{on}interval_mins = 2\n")), (true, 2));
        // No config file at all is the default config: off.
        std::fs::remove_file(&file).unwrap();
        let mut opened = false;
        auto_tick(&file, || {
            opened = true;
            bail!("unreachable")
        });
        assert!(!opened);
    }
}
