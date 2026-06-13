//! `mimir graph viz` — a self-contained interactive HTML view of the graph.
//!
//! Same local-first rules as the dashboard: no CDN, no webfonts, no JS
//! dependencies — one file, openable offline. The renderer is a hand-rolled
//! canvas force layout (Barnes–Hut) over the project's memories, symbols,
//! files, docs and communities, with the edges that tie them together.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use mimir_core::Mimir;
use serde_json::json;

pub fn viz(out: Option<String>, open: bool, max_nodes: usize) -> Result<()> {
    let mimir = Mimir::open()?;
    let proj = mimir
        .project_for_cwd(&std::env::current_dir()?)?
        .context("not inside a project (run from a repo, or pass --help)")?;
    let html = render(
        &mimir,
        proj.id,
        proj.title.as_deref().unwrap_or("project"),
        max_nodes,
    )?;
    let path = out.unwrap_or_else(|| {
        std::env::temp_dir()
            .join("mimir-graph.html")
            .to_string_lossy()
            .into_owned()
    });
    std::fs::write(&path, html).with_context(|| format!("write {path}"))?;
    println!("graph viz → {path}");
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

struct VizNode {
    id: i64,
    uid: String,
    kind: String,
    subkind: Option<String>,
    title: String,
    path: Option<String>,
    deg: i64,
    pinned: bool,
}

fn render(mimir: &Mimir, project_id: i64, project: &str, max_nodes: usize) -> Result<String> {
    let conn = &mimir.conn;

    // One pass over the edge table serves both the degree map here and the
    // included-set filtering at the end (it was two full scans before).
    let all_edges: Vec<(i64, i64, String)> = {
        let mut stmt = conn.prepare("SELECT src, dst, rel FROM edge")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let mut degree: HashMap<i64, i64> = HashMap::new();
    for (s, d, _) in &all_edges {
        *degree.entry(*s).or_default() += 1;
        *degree.entry(*d).or_default() += 1;
    }

    let fetch = |sql: &str, params: &[&dyn rusqlite::ToSql]| -> Result<Vec<VizNode>> {
        let mut stmt = conn.prepare(sql)?;
        let mut out = Vec::new();
        let mut rows = stmt.query(params)?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            out.push(VizNode {
                id,
                uid: r.get(1)?,
                kind: r.get(2)?,
                subkind: r.get(3)?,
                title: r
                    .get::<_, Option<String>>(4)?
                    .unwrap_or_else(|| "(untitled)".into()),
                path: r.get(5)?,
                deg: *degree.get(&id).unwrap_or(&0),
                pinned: r.get::<_, i64>(6)? != 0,
            });
        }
        Ok(out)
    };
    const COLS: &str = "id, uid, kind, subkind, title, path, pinned";
    let live = "deleted_at IS NULL AND superseded_by IS NULL";

    let mut nodes: Vec<VizNode> = Vec::new();
    let mut tag_edges: Vec<(i64, i64)> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut push_all = |batch: Vec<VizNode>, nodes: &mut Vec<VizNode>| {
        for n in batch {
            if seen.insert(n.id) {
                nodes.push(n);
            }
        }
    };

    // 1. The project node, its memories (incl. global), and its communities.
    push_all(
        fetch(
            &format!("SELECT {COLS} FROM node WHERE id = ?1 AND {live}"),
            &[&project_id],
        )?,
        &mut nodes,
    );
    push_all(
        fetch(
            &format!(
                "SELECT {COLS} FROM node WHERE kind='memory' AND {live}
                 AND (project_id = ?1 OR project_id IS NULL)"
            ),
            &[&project_id],
        )?,
        &mut nodes,
    );
    push_all(
        fetch(
            &format!(
                "SELECT {COLS} FROM node WHERE kind IN ('community','collection')
                 AND {live} AND (project_id = ?1 OR project_id IS NULL)"
            ),
            &[&project_id],
        )?,
        &mut nodes,
    );

    // 2. Anything a memory links to (code, docs, files) — these are the
    //    edges that make the unified graph worth looking at.
    let mem_ids: Vec<i64> = nodes
        .iter()
        .filter(|n| n.kind == "memory")
        .map(|n| n.id)
        .collect();
    if !mem_ids.is_empty() {
        // Chunked: the id list appears twice in the query, and SQLite caps
        // bound parameters at 32766 — a big store would abort otherwise.
        for chunk in mem_ids.chunks(8_000) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let sql = format!(
                "SELECT {COLS} FROM node WHERE {live} AND id IN (
                   SELECT dst FROM edge WHERE src IN ({placeholders})
                   UNION SELECT src FROM edge WHERE dst IN ({placeholders}))"
            );
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .chain(chunk.iter())
                .map(|id| id as &dyn rusqlite::ToSql)
                .collect();
            push_all(fetch(&sql, &params)?, &mut nodes);
        }
    }

    // 3. Fill the remaining budget with the project's best-connected symbols.
    let budget = max_nodes.saturating_sub(nodes.len());
    if budget > 0 {
        let mut symbols = fetch(
            &format!(
                "SELECT {COLS} FROM node WHERE kind='symbol' AND {live}
                 AND project_id = ?1"
            ),
            &[&project_id],
        )?;
        symbols.sort_by_key(|n| std::cmp::Reverse(n.deg));
        symbols.truncate(budget);
        push_all(symbols, &mut nodes);
    }

    // 4. Parent files of included symbols (gives the layout its skeleton).
    let parent_ids: Vec<i64> = {
        let included: Vec<i64> = nodes
            .iter()
            .filter(|n| n.kind == "symbol")
            .map(|n| n.id)
            .collect();
        if included.is_empty() {
            Vec::new()
        } else {
            let placeholders = vec!["?"; included.len()].join(",");
            let mut stmt = conn.prepare(&format!(
                "SELECT DISTINCT parent_id FROM node
                 WHERE id IN ({placeholders}) AND parent_id IS NOT NULL"
            ))?;
            let params: Vec<&dyn rusqlite::ToSql> = included
                .iter()
                .map(|id| id as &dyn rusqlite::ToSql)
                .collect();
            let mut rows = stmt.query(&params[..])?;
            let mut out = Vec::new();
            while let Some(r) = rows.next()? {
                out.push(r.get::<_, i64>(0)?);
            }
            out
        }
    };
    if !parent_ids.is_empty() {
        let placeholders = vec!["?"; parent_ids.len()].join(",");
        let sql = format!("SELECT {COLS} FROM node WHERE {live} AND id IN ({placeholders})");
        let params: Vec<&dyn rusqlite::ToSql> = parent_ids
            .iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();
        push_all(fetch(&sql, &params)?, &mut nodes);
    }

    // Virtual tag hubs (not stored): memories sharing a tag cluster by
    // topic even before anyone links them to code explicitly.
    {
        let mut tag_members: HashMap<String, Vec<i64>> = HashMap::new();
        let mem_set: HashSet<i64> = nodes
            .iter()
            .filter(|n| n.kind == "memory")
            .map(|n| n.id)
            .collect();
        let mut stmt = conn.prepare(
            "SELECT id, tags_text FROM node WHERE kind='memory'
             AND tags_text IS NOT NULL AND tags_text != ''",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            if !mem_set.contains(&id) {
                continue;
            }
            let tags: String = r.get(1)?;
            for tag in tags.split_whitespace() {
                tag_members.entry(tag.to_string()).or_default().push(id);
            }
        }
        let mut next_id = -1i64;
        let mut tags: Vec<_> = tag_members.into_iter().collect();
        tags.sort();
        for (tag, members) in tags {
            if members.len() < 2 {
                continue; // singleton tags add noise, not structure
            }
            let deg = members.len() as i64;
            nodes.push(VizNode {
                id: next_id,
                uid: format!("#{tag}"),
                kind: "tag".into(),
                subkind: None,
                title: format!("#{tag}"),
                path: None,
                deg,
                pinned: false,
            });
            for m in members {
                tag_edges.push((next_id, m));
            }
            next_id -= 1;
        }
    }

    // ---- edges among the included set (reuses the single edge scan) ----
    let included: HashSet<i64> = nodes.iter().map(|n| n.id).collect();
    let mut edges: Vec<(i64, i64, String)> = all_edges
        .into_iter()
        .filter(|(s, d, _)| included.contains(s) && included.contains(d))
        .collect();
    // symbol → parent file containment (implicit in parent_id, not in edge).
    {
        let mut stmt = conn.prepare(
            "SELECT id, parent_id FROM node WHERE kind='symbol' AND parent_id IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let (s, p): (i64, i64) = (r.get(0)?, r.get(1)?);
            if included.contains(&s) && included.contains(&p) {
                edges.push((p, s, "contains".into()));
            }
        }
    }
    for (tag, mem) in tag_edges {
        edges.push((tag, mem, "tagged".into()));
    }

    // ---- serialize ----
    let jnodes: Vec<serde_json::Value> = nodes
        .iter()
        .map(|n| {
            let mut title = n.title.clone();
            if title.len() > 96 {
                let mut cut = 96;
                while !title.is_char_boundary(cut) {
                    cut -= 1;
                }
                title.truncate(cut);
                title.push('…');
            }
            json!({
                "id": n.id, "uid": n.uid, "k": n.kind, "sk": n.subkind,
                "t": title, "p": n.path, "deg": n.deg, "pin": n.pinned,
            })
        })
        .collect();
    let jedges: Vec<serde_json::Value> =
        edges.iter().map(|(s, d, rel)| json!([s, d, rel])).collect();
    let data = json!({ "nodes": jnodes, "edges": jedges })
        .to_string()
        .replace("</", "<\\/");

    let html = TEMPLATE
        .replace("{{DATA}}", &data)
        .replace("{{PROJECT}}", &esc(project))
        .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
        .replace(
            "{{COUNTS}}",
            &format!("{} nodes · {} edges", nodes.len(), edges.len()),
        );
    Ok(html)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const TEMPLATE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mimir graph — {{PROJECT}}</title>
<style>
:root{--bg:#0b0e14;--panel:#10141c;--line:#232a38;--text:#e6e9ef;--dim:#8a93a6;
--mem:#e8a33d;--sym:#7ec699;--file:#5d7290;--doc:#5aa9e6;--com:#b48ead;--proj:#e6e9ef}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:var(--bg);
font:14px/1.5 system-ui,sans-serif;color:var(--text)}
canvas{position:absolute;inset:0;cursor:grab}
canvas.dragging{cursor:grabbing}
#hud{position:absolute;top:0;left:0;right:0;display:flex;gap:14px;align-items:center;
padding:12px 18px;pointer-events:none}
#hud>*{pointer-events:auto}
#hud .brand{font-weight:650;letter-spacing:.3px}
#hud .brand span{color:var(--mem)}
#hud input{background:var(--panel);border:1px solid var(--line);border-radius:8px;
color:var(--text);padding:7px 12px;width:260px;outline:none}
#hud input:focus{border-color:var(--dim)}
#legend{display:flex;gap:4px}
#legend button{background:var(--panel);border:1px solid var(--line);border-radius:20px;
color:var(--dim);padding:5px 12px;cursor:pointer;font-size:12.5px}
#legend button.on{color:var(--text)}
#legend button i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#legend button:not(.on) i{opacity:.25}
#meta{margin-left:auto;color:var(--dim);font-size:12.5px;text-align:right}
#tip{position:absolute;display:none;max-width:420px;background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13px;
pointer-events:none;box-shadow:0 12px 40px rgba(0,0,0,.5);z-index:3}
#tip .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
#tip .p{color:var(--dim);font-family:monospace;font-size:11.5px;word-break:break-all}
#side{position:absolute;top:64px;right:14px;bottom:14px;width:340px;background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:18px;display:none;overflow:auto}
#side h2{font-size:16px;margin-bottom:2px;word-break:break-word}
#side .k{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
#side .p{color:var(--dim);font-family:monospace;font-size:12px;word-break:break-all;margin:8px 0}
#side h3{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px;margin:16px 0 6px}
#side .nb{display:block;width:100%;text-align:left;background:none;border:none;color:var(--text);
padding:5px 6px;border-radius:6px;cursor:pointer;font-size:13px;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
#side .nb:hover{background:#161c28}
#side .nb i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}
#side .nb em{color:var(--dim);font-style:normal;font-size:11.5px;margin-right:6px}
#side .x{position:absolute;top:10px;right:14px;background:none;border:none;color:var(--dim);
font-size:18px;cursor:pointer}
#help{position:absolute;left:18px;bottom:14px;color:var(--dim);font-size:12px}
@media (prefers-reduced-motion: reduce){ /* layout settles before first paint */ }
</style></head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <div class="brand">mimir <span>graph</span>&nbsp;·&nbsp;{{PROJECT}}</div>
  <input id="q" placeholder="search nodes…" spellcheck="false">
  <div id="legend"></div>
  <div id="meta">{{COUNTS}}<br>v{{VERSION}}</div>
</div>
<div id="tip"></div>
<div id="side"><button class="x" onclick="deselect()">×</button><div id="sidebody"></div></div>
<div id="help">scroll&nbsp;zoom · drag&nbsp;pan · drag&nbsp;node&nbsp;to&nbsp;pin · click&nbsp;select · dbl-click&nbsp;release</div>
<script>
const DATA = {{DATA}};
const COLOR = {memory:'#e8a33d',symbol:'#7ec699',file:'#5d7290',chunk:'#5aa9e6',
  collection:'#5aa9e6',community:'#b48ead',project:'#e6e9ef',tag:'#d96c6c'};
const RELCOL = {calls:'rgba(126,198,153,.20)',imports:'rgba(126,198,153,.12)',
  contains:'rgba(93,114,144,.16)',member_of:'rgba(180,142,173,.25)',
  about:'rgba(232,163,61,.45)',relates:'rgba(232,163,61,.35)',
  mentions:'rgba(232,163,61,.30)',links:'rgba(232,163,61,.40)',
  tagged:'rgba(217,108,108,.22)',summarizes:'rgba(232,163,61,.35)'};
const N = DATA.nodes, E = DATA.edges;
const byId = new Map(N.map(n=>[n.id,n]));
// deterministic spirals, one per domain: code spawns west, knowledge east
// (the cross-domain link edges become the visible tissue between them).
function wellX(n){return (n.k==='symbol'||n.k==='file') ? -650
  : (n.k==='memory'||n.k==='tag'||n.k==='collection'||n.k==='chunk') ? 700 : 0;}
const spiral={};
N.forEach(n=>{
  const w=wellX(n), i=(spiral[w]=(spiral[w]||0)+1);
  const a=i*2.39996, r=20*Math.sqrt(i);
  n.x=w+Math.cos(a)*r; n.y=Math.sin(a)*r; n.vx=0; n.vy=0; n.fixed=false;
  n.r = (n.k==='project'?16 : n.k==='memory'?5.5 : 3.5) + Math.min(9, Math.sqrt(n.deg||0)*1.1);
  n.hidden=false;});
const adj = new Map();
E.forEach(([s,d])=>{ if(!adj.has(s))adj.set(s,new Set()); if(!adj.has(d))adj.set(d,new Set());
  adj.get(s).add(d); adj.get(d).add(s); });

// ---------- Barnes–Hut force layout ----------
function buildQuad(nodes){
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const n of nodes){x0=Math.min(x0,n.x);y0=Math.min(y0,n.y);x1=Math.max(x1,n.x);y1=Math.max(y1,n.y);}
  const s=Math.max(x1-x0,y1-y0)||1;
  const root={x:x0,y:y0,s:s,mass:0,cx:0,cy:0,kids:null,node:null};
  function insert(q,n){
    if(q.node===null && q.kids===null){q.node=n;return;}
    if(q.kids===null){q.kids=[null,null,null,null]; const old=q.node; q.node=null; place(q,old);}
    place(q,n);
  }
  function place(q,n){
    const h=q.s/2, ix=n.x>q.x+h?1:0, iy=n.y>q.y+h?1:0, i=iy*2+ix;
    if(!q.kids[i])q.kids[i]={x:q.x+ix*h,y:q.y+iy*h,s:h,mass:0,cx:0,cy:0,kids:null,node:null};
    insert(q.kids[i],n);
  }
  for(const n of nodes) insert(root,n);
  (function agg(q){ if(!q)return; let m=0,cx=0,cy=0;
    if(q.node){m=q.node.r;cx=q.node.x*m;cy=q.node.y*m;}
    if(q.kids)for(const k of q.kids){if(!k)continue;agg(k);m+=k.mass;cx+=k.cx*k.mass;cy+=k.cy*k.mass;}
    q.mass=m||1e-9; q.cx=m?cx/m:q.x; q.cy=m?cy/m:q.y;})(root);
  return root;
}
function repel(n,q,out){
  if(!q||q.mass<1e-8)return;
  const dx=n.x-q.cx, dy=n.y-q.cy, d2=dx*dx+dy*dy+0.01;
  if(q.node===n)return;
  if(q.node!==null || (q.s*q.s)/d2 < 0.64){
    const f=900*q.mass/d2; const d=Math.sqrt(d2);
    out[0]+=f*dx/d; out[1]+=f*dy/d; return;
  }
  if(q.kids)for(const k of q.kids)repel(n,k,out);
}
let alpha=1;
function tick(){
  const vis=N.filter(n=>!n.hidden);
  const quad=buildQuad(vis);
  for(const n of vis){
    if(n.fixed)continue;
    const f=[0,0]; repel(n,quad,f);
    // two gravity wells: code west, knowledge east — the link edges
    // between them become the visible "memory ↔ code" tissue.
    const gx = wellX(n);
    f[0]-=(n.x-gx)*0.015; f[1]-=n.y*0.012;
    n.vx=(n.vx+f[0]*alpha*0.02)*0.6; n.vy=(n.vy+f[1]*alpha*0.02)*0.6;
  }
  const RELK={tagged:0.008,contains:0.05,calls:0.035,imports:0.02,member_of:0.02,summarizes:0.03};
  const RELD={tagged:110,member_of:90};
  for(const [s,d,rel] of E){
    const a=byId.get(s), b=byId.get(d);
    if(!a||!b||a.hidden||b.hidden)continue;
    const dx=b.x-a.x, dy=b.y-a.y, dist=Math.sqrt(dx*dx+dy*dy)+0.01;
    const want=(RELD[rel]||46)+a.r+b.r;
    // soften springs on high-degree endpoints so hubs don't slingshot
    const soft=1/Math.sqrt(Math.min(adj.get(s)?.size||1, adj.get(d)?.size||1));
    const k=(dist-want)/dist*(RELK[rel]||0.04)*soft*alpha;
    if(!a.fixed){a.vx+=dx*k;a.vy+=dy*k;}
    if(!b.fixed){b.vx-=dx*k;b.vy-=dy*k;}
  }
  for(const n of vis){
    if(n.fixed)continue;
    const sp=Math.sqrt(n.vx*n.vx+n.vy*n.vy);
    if(sp>10){n.vx*=10/sp; n.vy*=10/sp;}     // velocity clamp: no slingshots
    n.x+=n.vx; n.y+=n.vy;
  }
  alpha=Math.max(0.002,alpha*0.995);
}

// ---------- rendering ----------
const canvas=document.getElementById('c'), ctx=canvas.getContext('2d');
let W,H,DPR; let cam={x:0,y:0,z:1};
function resize(){DPR=devicePixelRatio||1;W=innerWidth;H=innerHeight;
  canvas.width=W*DPR;canvas.height=H*DPR;canvas.style.width=W+'px';canvas.style.height=H+'px';}
addEventListener('resize',resize); resize();
let selected=null, hovered=null, query='';
function world(px,py){return [(px-W/2)/cam.z+cam.x,(py-H/2)/cam.z+cam.y];}
function draw(){
  ctx.setTransform(DPR,0,0,DPR,0,0); ctx.clearRect(0,0,W,H);
  ctx.save(); ctx.translate(W/2,H/2); ctx.scale(cam.z,cam.z); ctx.translate(-cam.x,-cam.y);
  const hl = selected ? adj.get(selected.id)||new Set() : null;
  for(const [s,d,rel] of E){
    const a=byId.get(s), b=byId.get(d);
    if(!a||!b||a.hidden||b.hidden)continue;
    let col=RELCOL[rel]||'rgba(138,147,166,.14)';
    if(selected){
      const on = s===selected.id||d===selected.id;
      col = on ? (RELCOL[rel]||'rgba(138,147,166,.5)').replace(/[\d.]+\)$/,'0.85)') : 'rgba(138,147,166,.04)';
    }
    ctx.strokeStyle=col; ctx.lineWidth=1/cam.z;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }
  for(const n of N){
    if(n.hidden)continue;
    const dim = (selected && n!==selected && !(hl&&hl.has(n.id))) ||
                (query && !n.t.toLowerCase().includes(query));
    ctx.globalAlpha = dim?0.12:1;
    ctx.fillStyle=COLOR[n.k]||'#8a93a6';
    ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,7); ctx.fill();
    if(n.pin){ctx.strokeStyle='#ffd98a';ctx.lineWidth=1.6/cam.z;
      ctx.beginPath();ctx.arc(n.x,n.y,n.r+2.5/cam.z,0,7);ctx.stroke();}
    if(n===selected||n===hovered){ctx.strokeStyle='#e6e9ef';ctx.lineWidth=1.6/cam.z;
      ctx.beginPath();ctx.arc(n.x,n.y,n.r+3.5/cam.z,0,7);ctx.stroke();}
    const showLabel = !dim && (n.r*cam.z>11 || n===selected || n===hovered ||
      (query && n.t.toLowerCase().includes(query)));
    if(showLabel){
      ctx.globalAlpha=dim?0.2:0.92; ctx.fillStyle='#c7cdd9';
      ctx.font=`${Math.max(11,12)/cam.z}px system-ui,sans-serif`;
      ctx.fillText(n.t.slice(0,42), n.x+n.r+4/cam.z, n.y+4/cam.z);
    }
  }
  ctx.globalAlpha=1;
  // domain labels floating above each cloud
  function cloudLabel(pred, big, small){
    let sx=0, minY=1e9, c=0;
    for(const n of N){ if(n.hidden||!pred(n))continue; sx+=n.x; minY=Math.min(minY,n.y-n.r); c++; }
    if(c<3)return;
    const x=sx/c;
    ctx.textAlign='center'; ctx.fillStyle='#3a4358';
    ctx.font=`650 ${30/cam.z}px system-ui,sans-serif`;
    ctx.fillText(big, x, minY-58/cam.z);
    ctx.fillStyle='#2c3445';
    ctx.font=`${15/cam.z}px system-ui,sans-serif`;
    ctx.fillText(small, x, minY-34/cam.z);
    ctx.textAlign='left';
  }
  cloudLabel(n=>n.k==='symbol'||n.k==='file', 'CODE', 'symbols · files · calls');
  cloudLabel(n=>n.k==='memory'||n.k==='tag'||n.k==='collection'||n.k==='chunk',
    'KNOWLEDGE', 'memories · docs · tags');
  ctx.restore();
}
(function loop(){ if(alpha>0.003)tick(); draw(); requestAnimationFrame(loop); })();
if(matchMedia('(prefers-reduced-motion: reduce)').matches){ for(let i=0;i<400;i++)tick(); alpha=0.002; }

// ---------- interaction ----------
let drag=null, panning=false, px=0, py=0;
function pick(mx,my){
  const [wx,wy]=world(mx,my); let best=null,bd=1e9;
  for(const n of N){ if(n.hidden)continue;
    const dx=n.x-wx,dy=n.y-wy,d=Math.sqrt(dx*dx+dy*dy);
    if(d<n.r+6/cam.z && d<bd){bd=d;best=n;} }
  return best;
}
canvas.addEventListener('mousedown',e=>{
  const n=pick(e.clientX,e.clientY);
  if(n){drag=n; n.fixed=true; alpha=Math.max(alpha,0.25);}
  else {panning=true; canvas.classList.add('dragging');}
  px=e.clientX; py=e.clientY;
});
addEventListener('mousemove',e=>{
  if(drag){const [wx,wy]=world(e.clientX,e.clientY); drag.x=wx; drag.y=wy; alpha=Math.max(alpha,0.18);}
  else if(panning){cam.x-=(e.clientX-px)/cam.z; cam.y-=(e.clientY-py)/cam.z;}
  else {hovered=pick(e.clientX,e.clientY); tip(e);}
  px=e.clientX; py=e.clientY;
});
addEventListener('mouseup',e=>{
  if(drag && Math.abs(e.clientX-px)<3 && Math.abs(e.clientY-py)<3){} // click handled below
  drag=null; panning=false; canvas.classList.remove('dragging');
});
canvas.addEventListener('click',e=>{
  const n=pick(e.clientX,e.clientY);
  if(n)select(n); else deselect();
});
canvas.addEventListener('dblclick',e=>{
  const n=pick(e.clientX,e.clientY); if(n){n.fixed=false; alpha=Math.max(alpha,0.3);}
});
canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  const f=Math.exp(-e.deltaY*0.0012), [wx,wy]=world(e.clientX,e.clientY);
  cam.z=Math.min(8,Math.max(0.08,cam.z*f));
  const [wx2,wy2]=world(e.clientX,e.clientY);
  cam.x+=wx-wx2; cam.y+=wy-wy2;
},{passive:false});
const tipEl=document.getElementById('tip');
function tip(e){
  if(!hovered){tipEl.style.display='none';return;}
  tipEl.style.display='block';
  tipEl.style.left=Math.min(W-440,e.clientX+16)+'px';
  tipEl.style.top=(e.clientY+16)+'px';
  tipEl.innerHTML=`<div class="k">${hovered.k}${hovered.sk?' · '+hovered.sk:''} · deg ${hovered.deg}</div>
    <div>${escapeHtml(hovered.t)}</div>${hovered.p?`<div class="p">${escapeHtml(hovered.p)}</div>`:''}`;
}
function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;');}
const side=document.getElementById('side'), sidebody=document.getElementById('sidebody');
function select(n){
  selected=n;
  const nbs=[...(adj.get(n.id)||[])].map(id=>byId.get(id)).filter(Boolean)
    .sort((a,b)=>b.deg-a.deg).slice(0,40);
  sidebody.innerHTML=`<div class="k">${n.k}${n.sk?' · '+n.sk:''} · ${n.uid}</div>
    <h2>${escapeHtml(n.t)}</h2>${n.p?`<div class="p">${escapeHtml(n.p)}</div>`:''}
    <h3>connected (${(adj.get(n.id)||new Set()).size})</h3>` +
    nbs.map(m=>`<button class="nb" data-id="${m.id}">
      <i style="background:${COLOR[m.k]||'#8a93a6'}"></i><em>${m.k}</em>${escapeHtml(m.t)}</button>`).join('');
  side.style.display='block';
  sidebody.querySelectorAll('.nb').forEach(b=>b.onclick=()=>{
    const m=byId.get(+b.dataset.id); if(m){select(m); cam.x=m.x; cam.y=m.y;}});
}
function deselect(){selected=null; side.style.display='none';}
addEventListener('keydown',e=>{if(e.key==='Escape'){deselect();document.getElementById('q').blur();}});
document.getElementById('q').addEventListener('input',e=>{query=e.target.value.trim().toLowerCase();});
// legend with kind toggles
const kinds=[...new Set(N.map(n=>n.k))].sort();
const leg=document.getElementById('legend');
for(const k of kinds){
  const b=document.createElement('button'); b.className='on';
  b.innerHTML=`<i style="background:${COLOR[k]||'#8a93a6'}"></i>${k} <span style="opacity:.5">${N.filter(n=>n.k===k).length}</span>`;
  b.onclick=()=>{const on=b.classList.toggle('on');
    N.forEach(n=>{if(n.k===k)n.hidden=!on;}); alpha=Math.max(alpha,0.3);};
  leg.appendChild(b);
}
</script>
</body></html>
"#;
