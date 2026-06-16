//! `mimir dashboard` — a self-contained HTML telemetry panel for the store.
//!
//! Local-first to the bone: no CDN, no webfonts, no JS dependencies —
//! one file you can open offline. Charts are hand-rolled inline SVG.

use anyhow::{Context, Result};
use mimir_core::model::now_unix;
use mimir_core::{learn, store, Mimir};

pub fn dashboard(out: Option<String>, open: bool) -> Result<()> {
    let mimir = Mimir::open()?;
    let html = render(&mimir)?;
    let path = out.unwrap_or_else(|| {
        std::env::temp_dir()
            .join("mimir-dashboard.html")
            .to_string_lossy()
            .into_owned()
    });
    crate::fsutil::write_private(std::path::Path::new(&path), &html)
        .with_context(|| format!("write {path}"))?;
    println!("dashboard → {path}");
    if open {
        let _ = std::process::Command::new(if cfg!(target_os = "macos") {
            "open"
        } else if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "xdg-open"
        })
        .args(if cfg!(target_os = "windows") {
            vec!["/c", "start", "", &path]
        } else {
            vec![path.as_str()]
        })
        .spawn();
    }
    Ok(())
}

struct Tile {
    label: &'static str,
    value: i64,
    unit: &'static str,
}

fn render(mimir: &Mimir) -> Result<String> {
    let conn = &mimir.conn;
    let now = now_unix();
    let q1 = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get(0))?) };

    // ---- headline tiles ----
    let memories =
        q1("SELECT count(*) FROM node WHERE kind='memory' AND deleted_at IS NULL AND superseded_by IS NULL")?;
    let chunks = q1("SELECT count(*) FROM node WHERE kind='chunk' AND deleted_at IS NULL")?;
    let symbols = q1("SELECT count(*) FROM node WHERE kind='symbol' AND deleted_at IS NULL")?;
    let edges = q1("SELECT count(*) FROM edge")?;
    let tiles = [
        Tile {
            label: "memories",
            value: memories,
            unit: "",
        },
        Tile {
            label: "doc chunks",
            value: chunks,
            unit: "",
        },
        Tile {
            label: "code symbols",
            value: symbols,
            unit: "",
        },
        Tile {
            label: "edges",
            value: edges,
            unit: "",
        },
    ];
    let projects = q1("SELECT count(*) FROM node WHERE kind='project' AND deleted_at IS NULL")?;
    let collections =
        q1("SELECT count(*) FROM node WHERE kind='collection' AND deleted_at IS NULL")?;
    let files = q1("SELECT count(*) FROM node WHERE kind='file' AND deleted_at IS NULL")?;

    // ---- memory types ----
    let mut stmt = conn.prepare(
        "SELECT coalesce(subkind,'note'), count(*) FROM node
         WHERE kind='memory' AND deleted_at IS NULL AND superseded_by IS NULL
         GROUP BY 1 ORDER BY 2 DESC",
    )?;
    let types: Vec<(String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    // ---- 14-day recall activity ----
    let day0 = now / 86_400 - 13;
    let mut stmt = conn.prepare(
        "SELECT at/86400, event, count(*) FROM recall_event
         WHERE at >= ?1 GROUP BY 1, 2",
    )?;
    let mut shown = [0i64; 14];
    let mut opened = [0i64; 14];
    for row in stmt.query_map([day0 * 86_400], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
        ))
    })? {
        let (day, event, n) = row?;
        let idx = (day - day0).clamp(0, 13) as usize;
        match event.as_str() {
            "shown" => shown[idx] += n,
            "opened" | "useful" => opened[idx] += n,
            _ => {}
        }
    }
    drop(stmt);

    // ---- effective-strength cohorts ----
    let mut stmt = conn.prepare(&format!(
        "SELECT {} FROM node WHERE kind='memory' AND deleted_at IS NULL AND superseded_by IS NULL",
        store::NODE_COLS
    ))?;
    let mems: Vec<mimir_core::model::Node> = stmt
        .query_map([], store::row_to_node)?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    let mut cohorts = [0i64; 5]; // <0.25 fading, <0.6, <1.2, <2.5, >=2.5 reinforced
    for m in &mems {
        let s = learn::effective_strength(m, now);
        let idx = if s < 0.25 {
            0
        } else if s < 0.6 {
            1
        } else if s < 1.2 {
            2
        } else if s < 2.5 {
            3
        } else {
            4
        };
        cohorts[idx] += 1;
    }

    // ---- most recalled ----
    let mut stmt = conn.prepare(
        "SELECT uid, coalesce(title,'(untitled)'), access_count FROM node
         WHERE deleted_at IS NULL AND access_count > 0 AND kind IN ('memory','symbol','file')
         ORDER BY access_count DESC, updated_at DESC LIMIT 6",
    )?;
    let top: Vec<(String, String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    // ---- per-project ----
    let mut stmt = conn.prepare(
        "SELECT coalesce(p.title,'global'), count(n.id) FROM node n
         LEFT JOIN node p ON p.id = n.project_id
         WHERE n.deleted_at IS NULL AND n.kind IN ('memory','chunk','symbol','file')
         GROUP BY 1 ORDER BY 2 DESC LIMIT 6",
    )?;
    let per_project: Vec<(String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);

    // ---- graph + hygiene + storage ----
    let calls_resolved =
        q1("SELECT count(*) FROM edge WHERE rel='calls' AND json_extract(meta,'$.resolved')=1")?;
    let calls_heuristic =
        q1("SELECT count(*) FROM edge WHERE rel='calls' AND json_extract(meta,'$.resolved')=0")?;
    let imports = q1("SELECT count(*) FROM edge WHERE rel='imports'")?;
    let links =
        q1("SELECT count(*) FROM edge WHERE rel IN ('about','relates','links','mentions')")?;
    let embedded = q1("SELECT count(*) FROM embedding")?;
    let embeddable = q1(
        "SELECT count(*) FROM node WHERE deleted_at IS NULL AND body IS NOT NULL
         AND kind IN ('memory','chunk','annotation','symbol')",
    )?;
    let superseded = q1("SELECT count(*) FROM node WHERE superseded_by IS NOT NULL")?;
    let archived =
        q1("SELECT count(*) FROM node WHERE deleted_at IS NOT NULL AND json_extract(meta,'$.archived')=1")?;
    let pinned = q1("SELECT count(*) FROM node WHERE pinned=1 AND deleted_at IS NULL")?;
    let events_total = q1("SELECT count(*) FROM recall_event")?;
    let db_bytes = std::fs::metadata(&mimir.paths.db_file)?.len();

    // ================= fragments =================
    let tiles_html: String = tiles
        .iter()
        .map(|t| {
            format!(
                r#"<div class="tile"><div class="tile-v">{}{}</div><div class="tile-l">{}</div></div>"#,
                fmt_n(t.value),
                t.unit,
                t.label
            )
        })
        .collect();

    // Well ripples: area-proportional concentric rings (memories innermost).
    let comp = [
        ("memories", memories, "var(--teal)"),
        ("doc chunks", chunks, "var(--teal-dim)"),
        ("code symbols", symbols, "var(--navy-bright)"),
    ];
    let total: i64 = comp.iter().map(|c| c.1).sum::<i64>().max(1);
    let mut acc = 0f64;
    let mut rings = String::new();
    let mut legend = String::new();
    for (i, (label, value, color)) in comp.iter().enumerate() {
        acc += *value as f64 / total as f64;
        let r = 24.0 + 64.0 * acc.sqrt();
        rings.push_str(&format!(
            r#"<circle class="ripple" style="animation-delay:{}ms" cx="110" cy="110" r="{:.1}" fill="none" stroke="{}" stroke-width="{}"/>"#,
            200 + i * 180,
            r,
            color,
            if i == 0 { 2.5 } else { 1.5 }
        ));
        legend.push_str(&format!(
            r#"<div class="lg"><span class="dot" style="background:{}"></span>{} <b>{}</b></div>"#,
            color,
            label,
            fmt_n(*value)
        ));
    }

    // Activity bars (overlaid shown/opened)
    let max_act = shown
        .iter()
        .chain(opened.iter())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    let mut act_bars = String::new();
    for i in 0..14 {
        let x = i as f64 * 22.0;
        let hs = 72.0 * shown[i] as f64 / max_act as f64;
        let ho = 72.0 * opened[i] as f64 / max_act as f64;
        act_bars.push_str(&format!(
            r#"<rect class="grow" style="animation-delay:{}ms" x="{:.0}" y="{:.1}" width="9" height="{:.1}" fill="var(--navy-bright)"/><rect class="grow" style="animation-delay:{}ms" x="{:.0}" y="{:.1}" width="9" height="{:.1}" fill="var(--teal)"/>"#,
            i * 40, x, 76.0 - hs, hs.max(1.0),
            i * 40 + 20, x + 10.0, 76.0 - ho, ho.max(1.0),
        ));
    }

    // Memory-type horizontal bars
    let max_t = types.first().map(|t| t.1).unwrap_or(1).max(1);
    let type_rows: String = types
        .iter()
        .map(|(name, n)| {
            format!(
                r#"<div class="hrow"><span class="hlabel">{}</span><span class="hbar"><i class="grow-x" style="width:{:.1}%"></i></span><span class="hval">{}</span></div>"#,
                name,
                100.0 * *n as f64 / max_t as f64,
                fmt_n(*n)
            )
        })
        .collect();

    // Strength cohorts
    let cohort_labels = ["fading", "waning", "steady", "strong", "reinforced"];
    let max_c = cohorts.iter().copied().max().unwrap_or(1).max(1);
    let cohort_cols: String = cohorts
        .iter()
        .zip(cohort_labels)
        .enumerate()
        .map(|(i, (n, label))| {
            let h = 64.0 * *n as f64 / max_c as f64;
            let color = if i == 0 { "var(--gold)" } else { "var(--teal)" };
            format!(
                r#"<div class="ccol"><div class="cbarwrap"><div class="cbar grow" style="height:{:.0}px;background:{};animation-delay:{}ms"></div></div><div class="clabel">{}</div><div class="cval">{}</div></div>"#,
                h.max(2.0), color, i * 90, label, fmt_n(*n)
            )
        })
        .collect();

    let top_rows: String = if top.is_empty() {
        r#"<div class="empty">no recalls yet — the well is undisturbed</div>"#.into()
    } else {
        top.iter()
            .map(|(uid, title, n)| {
                let tail = &uid[uid.len().saturating_sub(6)..];
                format!(
                    r#"<div class="trow"><span class="tid">{}</span><span class="ttitle">{}</span><span class="tcount">↑{}</span></div>"#,
                    tail,
                    esc(&truncate(title, 58)),
                    n
                )
            })
            .collect()
    };

    let max_p = per_project.first().map(|p| p.1).unwrap_or(1).max(1);
    let project_rows: String = per_project
        .iter()
        .map(|(name, n)| {
            format!(
                r#"<div class="hrow"><span class="hlabel">{}</span><span class="hbar"><i class="grow-x" style="width:{:.1}%"></i></span><span class="hval">{}</span></div>"#,
                esc(&truncate(name, 18)),
                100.0 * *n as f64 / max_p as f64,
                fmt_n(*n)
            )
        })
        .collect();

    let coverage = if embeddable > 0 {
        100.0 * embedded as f64 / embeddable as f64
    } else {
        0.0
    };

    // ---- token savings (outline/peek/filter/proxy) ----
    let saved_all = mimir_core::savings::totals(conn, None)?;
    let saved_by_source = mimir_core::savings::by_source(conn, None)?;
    let max_sv = saved_by_source
        .iter()
        .map(|(_, t)| t.saved())
        .max()
        .unwrap_or(1)
        .max(1);
    let savings_rows: String = if saved_by_source.is_empty() {
        r#"<div class="empty">no token savings recorded yet — try outline/peek or `mimir run`</div>"#.into()
    } else {
        saved_by_source
            .iter()
            .map(|(src, t)| {
                format!(
                    r#"<div class="hrow"><span class="hlabel">{}</span><span class="hbar"><i class="grow-x" style="width:{:.1}%"></i></span><span class="hval">{}</span></div>"#,
                    esc(src),
                    100.0 * t.saved() as f64 / max_sv as f64,
                    fmt_n(t.saved())
                )
            })
            .collect()
    };

    let html = TEMPLATE
        .replace("{{GENERATED}}", &mimir_core::format::full_date(now))
        .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
        .replace(
            "{{DB_PATH}}",
            &esc(&mimir.paths.db_file.display().to_string()),
        )
        .replace("{{DB_SIZE}}", &fmt_bytes(db_bytes))
        .replace("{{TILES}}", &tiles_html)
        .replace("{{RINGS}}", &rings)
        .replace("{{LEGEND}}", &legend)
        .replace("{{ACT_BARS}}", &act_bars)
        .replace(
            "{{DAY_FIRST}}",
            &mimir_core::format::month_day(day0 * 86_400),
        )
        .replace("{{DAY_LAST}}", &mimir_core::format::month_day(now))
        .replace("{{TYPE_ROWS}}", &type_rows)
        .replace("{{COHORTS}}", &cohort_cols)
        .replace("{{TOP_ROWS}}", &top_rows)
        .replace("{{PROJECT_ROWS}}", &project_rows)
        .replace("{{PROJECTS}}", &fmt_n(projects))
        .replace("{{COLLECTIONS}}", &fmt_n(collections))
        .replace("{{FILES}}", &fmt_n(files))
        .replace("{{CALLS_R}}", &fmt_n(calls_resolved))
        .replace("{{CALLS_H}}", &fmt_n(calls_heuristic))
        .replace("{{IMPORTS}}", &fmt_n(imports))
        .replace("{{LINKS}}", &fmt_n(links))
        .replace("{{EMBEDDED}}", &fmt_n(embedded))
        .replace("{{COVERAGE}}", &format!("{coverage:.0}"))
        .replace("{{MODEL}}", &esc(&mimir.config.embedding.model))
        .replace("{{SUPERSEDED}}", &fmt_n(superseded))
        .replace("{{ARCHIVED}}", &fmt_n(archived))
        .replace("{{ARCHIVED_CLASS}}", if archived > 0 { "gold" } else { "" })
        .replace("{{PINNED}}", &fmt_n(pinned))
        .replace("{{EVENTS}}", &fmt_n(events_total))
        .replace("{{SAVED_TOTAL}}", &fmt_n(saved_all.saved()))
        .replace("{{SAVED_BEFORE}}", &fmt_n(saved_all.before))
        .replace("{{SAVED_EVENTS}}", &fmt_n(saved_all.events))
        .replace(
            "{{SAVED_USD}}",
            &format!(
                "${:.2}",
                mimir_core::savings::to_dollars(
                    saved_all.saved(),
                    mimir.config.savings.input_price_per_mtok
                )
            ),
        )
        .replace("{{SAVINGS_ROWS}}", &savings_rows);
    Ok(html)
}

fn fmt_n(n: i64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push('\u{2009}'); // thin space
        }
        out.push(c);
    }
    out
}

fn fmt_bytes(b: u64) -> String {
    if b > 1_048_576 * 100 {
        format!("{:.1} GB", b as f64 / 1_073_741_824.0)
    } else if b > 10_240 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{b} B")
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}

const TEMPLATE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mimir — the well</title>
<style>
:root{
  --bg:#0a0e13; --panel:#0e151d; --line:#1c2936; --line-soft:#15202b;
  --ink:#c9d7e2; --dim:#5d7287; --faint:#3a4c5e;
  --teal:#1fb6a8; --teal-dim:#157f78; --navy-bright:#4e8cc7; --gold:#d9a441;
}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--bg)}
body{
  font-family:"Berkeley Mono","JetBrains Mono","IBM Plex Mono","Cascadia Code",ui-monospace,Menlo,monospace;
  color:var(--ink); min-height:100vh; padding:40px 24px 64px;
  background:
    radial-gradient(1100px 500px at 70% -10%, rgba(31,182,168,.05), transparent 60%),
    repeating-linear-gradient(0deg, transparent 0 2px, rgba(255,255,255,.012) 2px 4px),
    var(--bg);
}
.wrap{max-width:1180px;margin:0 auto}
header{display:flex;align-items:baseline;gap:18px;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px;flex-wrap:wrap}
.mark{font-size:22px;font-weight:700;letter-spacing:.42em;color:var(--ink)}
.mark b{color:var(--teal);font-weight:700}
.sub{color:var(--dim);font-size:12px;letter-spacing:.14em;text-transform:uppercase}
.meta{margin-left:auto;color:var(--faint);font-size:11.5px;text-align:right;line-height:1.7}
.meta b{color:var(--dim);font-weight:400}

.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px}
.tile{position:relative;background:var(--panel);border:1px solid var(--line);padding:18px 20px 14px}
.tile-v{font-size:34px;font-weight:700;letter-spacing:.02em;color:var(--ink);font-variant-numeric:tabular-nums}
.tile-l{color:var(--dim);font-size:11px;letter-spacing:.22em;text-transform:uppercase;margin-top:6px}
.tile::before,.panel::before{content:"";position:absolute;top:-1px;left:-1px;width:9px;height:9px;border-top:1px solid var(--teal);border-left:1px solid var(--teal);opacity:.85}
.tile::after,.panel::after{content:"";position:absolute;bottom:-1px;right:-1px;width:9px;height:9px;border-bottom:1px solid var(--teal);border-right:1px solid var(--teal);opacity:.35}

.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.panel{position:relative;background:var(--panel);border:1px solid var(--line);padding:18px 20px}
.s4{grid-column:span 4}.s5{grid-column:span 5}.s6{grid-column:span 6}.s7{grid-column:span 7}.s8{grid-column:span 8}.s12{grid-column:span 12}
h2{font-size:11px;font-weight:700;letter-spacing:.26em;text-transform:uppercase;color:var(--dim);margin-bottom:16px}
h2::before{content:"◈ ";color:var(--teal)}

.well{display:flex;gap:18px;align-items:center}
.well svg{flex:none}
.ripple{transform-box:fill-box;transform-origin:center;animation:ripple .9s cubic-bezier(.2,.8,.2,1) backwards}
@keyframes ripple{from{transform:scale(.55);opacity:0}to{transform:scale(1);opacity:1}}
.well-eye{fill:var(--teal);opacity:.9}
.lg{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:12.5px;margin:7px 0}
.lg b{color:var(--ink);margin-left:auto;font-variant-numeric:tabular-nums}
.legend{flex:1;min-width:0}
.dot{width:8px;height:8px;flex:none;display:inline-block}

.grow{transform-box:fill-box;transform-origin:bottom;animation:grow .7s cubic-bezier(.2,.8,.2,1) backwards}
@keyframes grow{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.axis{color:var(--faint);font-size:10px}
.key{display:flex;gap:16px;color:var(--dim);font-size:11px;margin-top:10px}
.key i{width:8px;height:8px;display:inline-block;margin-right:6px}

.hrow{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:12.5px}
.hlabel{width:92px;flex:none;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hbar{flex:1;height:10px;background:var(--line-soft);position:relative;overflow:hidden}
.hbar i{position:absolute;inset:0 auto 0 0;background:var(--teal)}
.grow-x{animation:growx .8s cubic-bezier(.2,.8,.2,1) backwards}
@keyframes growx{from{width:0!important}}
.hval{width:56px;text-align:right;color:var(--ink);font-variant-numeric:tabular-nums}

.cohorts{display:flex;gap:10px;align-items:flex-end}
.ccol{flex:1;text-align:center}
.cbarwrap{height:64px;display:flex;align-items:flex-end;justify-content:center}
.cbar{width:60%;}
.clabel{color:var(--dim);font-size:10px;letter-spacing:.08em;margin-top:8px;text-transform:uppercase}
.cval{color:var(--ink);font-size:13px;margin-top:2px;font-variant-numeric:tabular-nums}

.trow{display:flex;gap:12px;align-items:baseline;padding:7px 0;border-bottom:1px dashed var(--line-soft);font-size:12.5px}
.trow:last-child{border-bottom:none}
.tid{color:var(--teal);flex:none}
.ttitle{color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tcount{margin-left:auto;color:var(--gold);flex:none}
.empty{color:var(--faint);font-size:12.5px;padding:14px 0;font-style:italic}

.kv{display:grid;grid-template-columns:1fr 1fr;gap:7px 24px;font-size:12.5px}
.kv div{display:flex;justify-content:space-between;border-bottom:1px dashed var(--line-soft);padding:5px 0}
.kv span{color:var(--dim)} .kv b{color:var(--ink);font-weight:400;font-variant-numeric:tabular-nums}
.kv b.gold{color:var(--gold)} .kv b.teal{color:var(--teal)}

footer{margin-top:26px;border-top:1px solid var(--line);padding-top:14px;display:flex;color:var(--faint);font-size:11px;letter-spacing:.18em}
footer span{margin-left:auto}
@media (prefers-reduced-motion:reduce){.ripple,.grow,.grow-x{animation:none}}
@media (max-width:900px){.tiles{grid-template-columns:repeat(2,1fr)}.grid>*{grid-column:span 12}}
</style></head><body><div class="wrap">

<header>
  <div class="mark">MIMI<b>R</b></div>
  <div class="sub">memory engine · telemetry</div>
  <div class="meta">v{{VERSION}} · {{GENERATED}}<br><b>{{DB_PATH}}</b> · {{DB_SIZE}}</div>
</header>

<div class="tiles">{{TILES}}</div>

<div class="grid">
  <div class="panel s5">
    <h2>The well — store composition</h2>
    <div class="well">
      <svg width="220" height="220" viewBox="0 0 220 220">
        <circle cx="110" cy="110" r="97" fill="none" stroke="var(--line)" stroke-width="1"/>
        <circle cx="110" cy="110" r="3.5" class="well-eye"/>
        {{RINGS}}
      </svg>
      <div class="legend">{{LEGEND}}
        <div class="lg"><span class="dot" style="background:var(--faint)"></span>projects <b>{{PROJECTS}}</b></div>
        <div class="lg"><span class="dot" style="background:var(--faint)"></span>collections <b>{{COLLECTIONS}}</b></div>
        <div class="lg"><span class="dot" style="background:var(--faint)"></span>files <b>{{FILES}}</b></div>
      </div>
    </div>
  </div>

  <div class="panel s7">
    <h2>Recall activity — 14 days</h2>
    <svg width="100%" height="96" viewBox="0 0 308 96" preserveAspectRatio="none">
      <line x1="0" y1="76.5" x2="308" y2="76.5" stroke="var(--line)" stroke-width="1"/>
      {{ACT_BARS}}
      <text x="0" y="92" class="axis" fill="var(--faint)" font-size="9" font-family="inherit">{{DAY_FIRST}}</text>
      <text x="308" y="92" text-anchor="end" class="axis" fill="var(--faint)" font-size="9" font-family="inherit">{{DAY_LAST}}</text>
    </svg>
    <div class="key"><span><i style="background:var(--navy-bright)"></i>shown in results</span>
    <span><i style="background:var(--teal)"></i>opened / marked useful</span></div>
  </div>

  <div class="panel s4">
    <h2>Memory types</h2>
    {{TYPE_ROWS}}
  </div>

  <div class="panel s4">
    <h2>Strength cohorts</h2>
    <div class="cohorts">{{COHORTS}}</div>
  </div>

  <div class="panel s4">
    <h2>Most recalled</h2>
    {{TOP_ROWS}}
  </div>

  <div class="panel s6">
    <h2>Knowledge by project</h2>
    {{PROJECT_ROWS}}
  </div>

  <div class="panel s12">
    <h2>Token savings — dense context &amp; filtered output</h2>
    <div class="kv">
      <div><span>tokens saved</span><b class="teal">{{SAVED_TOTAL}}</b></div>
      <div><span>≈ saved</span><b class="gold">{{SAVED_USD}}</b></div>
      <div><span>source tokens seen</span><b>{{SAVED_BEFORE}}</b></div>
      <div><span>savings events</span><b>{{SAVED_EVENTS}}</b></div>
    </div>
    {{SAVINGS_ROWS}}
  </div>

  <div class="panel s6">
    <h2>Graph &amp; hygiene</h2>
    <div class="kv">
      <div><span>calls resolved</span><b class="teal">{{CALLS_R}}</b></div>
      <div><span>calls heuristic</span><b>{{CALLS_H}}</b></div>
      <div><span>import edges</span><b>{{IMPORTS}}</b></div>
      <div><span>memory↔code links</span><b class="teal">{{LINKS}}</b></div>
      <div><span>embedded</span><b>{{EMBEDDED}} ({{COVERAGE}}%)</b></div>
      <div><span>model</span><b>{{MODEL}}</b></div>
      <div><span>superseded</span><b>{{SUPERSEDED}}</b></div>
      <div><span>archived</span><b class="{{ARCHIVED_CLASS}}">{{ARCHIVED}}</b></div>
      <div><span>pinned</span><b>{{PINNED}}</b></div>
      <div><span>recall events</span><b>{{EVENTS}}</b></div>
    </div>
  </div>
</div>

<footer>ᛗᛁᛗᛁᚱ — what is remembered, remains<span>local · private · yours</span></footer>
</div></body></html>
"##;
