use anyhow::Result;
use mimir_core::config::{Config, Paths};
use mimir_core::{db, store, Mimir};

pub fn init() -> Result<()> {
    let paths = Paths::resolve()?;
    let config = Config::load(&paths.config_file)?;
    config.save(&paths.config_file)?;
    // Opening creates + migrates the database.
    let _conn = db::open(&paths.db_file)?;
    println!("config  {}", paths.config_file.display());
    println!("db      {}", paths.db_file.display());
    println!();
    println!("Register the MCP server once, globally:");
    println!("  claude mcp add --scope user mimir -- mimir mcp");
    Ok(())
}

pub fn status(json: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let counts = store::count_by_kind(&mimir.conn)?;
    let db_size = std::fs::metadata(&mimir.paths.db_file)
        .map(|m| m.len())
        .unwrap_or(0);
    let project = mimir.project_for_cwd(&std::env::current_dir()?)?;

    if json {
        let counts_json: serde_json::Map<String, serde_json::Value> = counts
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::json!(v)))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "db": mimir.paths.db_file,
                "db_bytes": db_size,
                "project": project.as_ref().and_then(|p| p.title.clone()),
                "counts": counts_json,
            })
        );
        return Ok(());
    }

    match &project {
        Some(p) => println!(
            "project {} ({})",
            p.title.as_deref().unwrap_or("?"),
            p.path.as_deref().unwrap_or("?")
        ),
        None => println!("project (none — global scope)"),
    }
    if counts.is_empty() {
        println!("store   empty");
    } else {
        let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
        println!("store   {}", summary.join(", "));
    }
    println!(
        "db      {} ({} KB)",
        mimir.paths.db_file.display(),
        db_size / 1024
    );
    Ok(())
}

pub fn doctor() -> Result<()> {
    let paths = Paths::resolve()?;
    let mut failures = 0;

    let check = |name: &str, ok: bool, detail: String, failures: &mut i32| {
        let mark = if ok { "ok " } else { "FAIL" };
        if !ok {
            *failures += 1;
        }
        println!("{mark}  {name}: {detail}");
    };

    match db::open(&paths.db_file) {
        Ok(conn) => {
            check(
                "db",
                true,
                paths.db_file.display().to_string(),
                &mut failures,
            );
            let integrity: String = conn
                .query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .unwrap_or_else(|e| format!("error: {e}"));
            check("integrity", integrity == "ok", integrity, &mut failures);
            let fts = conn
                .prepare("SELECT count(*) FROM node_fts")
                .and_then(|mut s| s.query_row([], |r| r.get::<_, i64>(0)));
            check(
                "fts5",
                fts.is_ok(),
                fts.map(|n| format!("{n} rows indexed"))
                    .unwrap_or_else(|e| e.to_string()),
                &mut failures,
            );
        }
        Err(e) => check("db", false, e.to_string(), &mut failures),
    }

    let model_present = paths.models_dir.exists()
        && std::fs::read_dir(&paths.models_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
    check(
        "model",
        true, // informational until embeddings land; BM25-only is a valid state
        if model_present {
            format!("present at {}", paths.models_dir.display())
        } else {
            "not downloaded (search is BM25-only until `mimir init` fetches it)".into()
        },
        &mut failures,
    );

    if failures > 0 {
        anyhow::bail!("{failures} check(s) failed");
    }
    Ok(())
}
