//! `mimir proxy` — run the optional local API proxy (own tokio runtime).

use anyhow::{Context, Result};
use mimir_core::Mimir;

pub fn run(
    bind: Option<String>,
    dry_run: bool,
    no_cache: bool,
    no_dedup: bool,
    prune: bool,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let pc = &mimir.config.proxy;
    let bind_str = bind.unwrap_or_else(|| pc.bind.clone());
    let addr = bind_str
        .parse()
        .with_context(|| format!("invalid --bind address: {bind_str}"))?;

    let cfg = mimir_proxy::ProxyConfig {
        bind: addr,
        upstream: pc.upstream.clone(),
        dry_run,
        cache: pc.cache && !no_cache,
        dedup: pc.dedup && !no_dedup,
        prune: prune || pc.prune,
        db_path: Some(mimir.paths.db_file.clone()),
    };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(mimir_proxy::serve(cfg))
}
