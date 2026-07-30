//! `mimir proxy` — run the optional local API proxy (own tokio runtime).

use anyhow::{anyhow, Context, Result};
use mimir_core::Mimir;
use mimir_proxy::CacheTtl;

pub fn run(
    bind: Option<String>,
    dry_run: bool,
    no_cache: bool,
    cache_ttl: Option<String>,
    no_dedup: bool,
    prune: bool,
) -> Result<()> {
    let mimir = Mimir::open()?;
    let pc = &mimir.config.proxy;
    let bind_str = bind.unwrap_or_else(|| pc.bind.clone());
    let addr = bind_str
        .parse()
        .with_context(|| format!("invalid --bind address: {bind_str}"))?;

    // Reject an unknown TTL rather than silently forwarding 5m: a user who
    // asked for 1h and got 5m would see no error and a bill they can't explain.
    let ttl_str = cache_ttl.as_deref().unwrap_or(pc.cache_ttl.as_str());
    let ttl = CacheTtl::parse(ttl_str).ok_or_else(|| {
        anyhow!("invalid cache TTL {ttl_str:?}: expected \"5m\" or \"1h\" (see [proxy] cache_ttl)")
    })?;

    let cfg = mimir_proxy::ProxyConfig {
        bind: addr,
        upstream: pc.upstream.clone(),
        dry_run,
        cache: pc.cache && !no_cache,
        cache_ttl: ttl,
        dedup: pc.dedup && !no_dedup,
        prune: prune || pc.prune,
        db_path: Some(mimir.paths.db_file.clone()),
        event_retention_days: mimir.config.learn.effective_retention_days(),
    };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(mimir_proxy::serve(cfg))
}
