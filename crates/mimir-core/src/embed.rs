//! Local ONNX embeddings via fastembed — lazy, optional, never required.
//!
//! The model downloads only on explicit request (`mimir init` /
//! `mimir embed --fetch`); a marker file gates silent loads so an agent
//! call can never block on a network fetch. Without a model everything
//! degrades to BM25-only.

use std::path::PathBuf;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::{params, Connection, OptionalExtension};

use crate::config::Paths;
use crate::error::{Error, Result};

pub const EMBED_BATCH: usize = 64;

/// Kinds whose body participates in semantic recall.
const EMBEDDABLE_KINDS: &str = "'memory', 'chunk', 'annotation', 'symbol'";

pub fn model_from_name(name: &str) -> Result<EmbeddingModel> {
    Ok(match name {
        // Quantized ONNX port: 384-dim, ~34 MB — the default.
        "bge-small-en-v1.5" | "bge-small-en-v1.5-q" => EmbeddingModel::BGESmallENV15Q,
        "bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "all-minilm-l6-v2" => EmbeddingModel::AllMiniLML6V2,
        other => {
            return Err(Error::Invalid(format!(
                "unknown embedding model '{other}' (try bge-small-en-v1.5)"
            )))
        }
    })
}

/// Marker written after the first successful model load; its presence is
/// what allows silent (non-downloading) loads later.
fn marker_path(paths: &Paths, name: &str) -> PathBuf {
    paths.models_dir.join(format!(".{name}.ready"))
}

pub fn model_ready(paths: &Paths, name: &str) -> bool {
    marker_path(paths, name).exists()
}

pub struct Embedder {
    model: TextEmbedding,
    pub name: String,
    pub dim: usize,
}

impl Embedder {
    /// Load the model. With `allow_download = false` this refuses unless a
    /// previous load completed (marker present), so it never touches the
    /// network on an agent's hot path.
    pub fn load(paths: &Paths, name: &str, allow_download: bool) -> Result<Embedder> {
        if !allow_download && !model_ready(paths, name) {
            return Err(Error::Invalid(format!(
                "embedding model '{name}' not downloaded; run `mimir init` or `mimir embed --fetch`"
            )));
        }
        let which = model_from_name(name)?;
        std::fs::create_dir_all(&paths.models_dir).map_err(|e| Error::io(&paths.models_dir, e))?;
        let options = TextInitOptions::new(which)
            .with_cache_dir(paths.models_dir.clone())
            .with_show_download_progress(allow_download);
        let mut model = TextEmbedding::try_new(options).map_err(|e| Error::Embed(e.to_string()))?;
        let probe = model
            .embed(["dimension probe"], None)
            .map_err(|e| Error::Embed(e.to_string()))?;
        let dim = probe.first().map(|v| v.len()).unwrap_or(0);
        if dim == 0 {
            return Err(Error::Embed("model returned empty embedding".into()));
        }
        std::fs::write(marker_path(paths, name), b"ok")
            .map_err(|e| Error::io(&paths.models_dir, e))?;
        Ok(Embedder {
            model,
            name: name.to_string(),
            dim,
        })
    }

    /// Embed texts, L2-normalized (so cosine == dot product downstream).
    pub fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut out = self
            .model
            .embed(&texts, Some(EMBED_BATCH))
            .map_err(|e| Error::Embed(e.to_string()))?;
        for v in &mut out {
            l2_normalize(v);
        }
        Ok(out)
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn from_blob(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Embed every node whose content is new or changed for this model.
/// Unchanged content never reaches the model: embeddings are keyed by
/// content hash, and identical text elsewhere in the store is copied.
pub fn embed_pending(conn: &Connection, embedder: &mut Embedder) -> Result<usize> {
    struct Pending {
        id: i64,
        text: String,
        hash: Vec<u8>,
    }
    let mut stmt = conn.prepare(&format!(
        "SELECT n.id, n.title, n.body, n.content_hash
         FROM node n LEFT JOIN embedding e ON e.node_id = n.id
         WHERE n.deleted_at IS NULL AND n.body IS NOT NULL
           AND n.kind IN ({EMBEDDABLE_KINDS})
           AND (e.node_id IS NULL OR e.model <> ?1
                OR n.content_hash IS NULL OR e.content_hash <> n.content_hash)"
    ))?;
    let pending: Vec<Pending> = stmt
        .query_map([&embedder.name], |r| {
            let title: Option<String> = r.get(1)?;
            let body: String = r.get(2)?;
            let hash: Option<Vec<u8>> = r.get(3)?;
            let text = match &title {
                Some(t) if !t.is_empty() => format!("{t}\n{body}"),
                _ => body.clone(),
            };
            Ok(Pending {
                id: r.get(0)?,
                hash: hash.unwrap_or_else(|| blake3::hash(body.as_bytes()).as_bytes().to_vec()),
                text,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    drop(stmt);
    if pending.is_empty() {
        return Ok(0);
    }

    let tx = conn.unchecked_transaction()?;
    let mut to_embed: Vec<&Pending> = Vec::new();
    for p in &pending {
        // Same content already embedded under another node? Copy it.
        let existing: Option<(i64, Vec<u8>)> = tx
            .query_row(
                "SELECT dim, vec FROM embedding
                 WHERE model = ?1 AND content_hash = ?2 AND node_id <> ?3 LIMIT 1",
                params![embedder.name, p.hash, p.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        match existing {
            Some((dim, vec)) => {
                upsert(&tx, p.id, &embedder.name, dim as usize, &vec, &p.hash)?;
            }
            None => to_embed.push(p),
        }
    }
    if !to_embed.is_empty() {
        let texts: Vec<String> = to_embed.iter().map(|p| p.text.clone()).collect();
        let vectors = embedder.embed(texts)?;
        for (p, vec) in to_embed.iter().zip(&vectors) {
            upsert(&tx, p.id, &embedder.name, vec.len(), &to_blob(vec), &p.hash)?;
        }
    }
    tx.commit()?;
    Ok(pending.len())
}

fn upsert(
    conn: &Connection,
    node_id: i64,
    model: &str,
    dim: usize,
    vec_blob: &[u8],
    hash: &[u8],
) -> Result<()> {
    conn.execute(
        "INSERT INTO embedding (node_id, model, dim, vec, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(node_id) DO UPDATE SET
           model = excluded.model, dim = excluded.dim,
           vec = excluded.vec, content_hash = excluded.content_hash",
        params![node_id, model, dim as i64, vec_blob, hash],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trip() {
        let v = vec![0.25f32, -1.5, 3.0e-7, 42.0];
        assert_eq!(from_blob(&to_blob(&v)), v);
    }

    #[test]
    fn normalize() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        let mut zero = vec![0.0f32, 0.0];
        l2_normalize(&mut zero); // must not NaN
        assert_eq!(zero, vec![0.0, 0.0]);
    }

    #[test]
    fn model_names() {
        assert!(model_from_name("bge-small-en-v1.5").is_ok());
        assert!(model_from_name("nonsense").is_err());
    }
}
