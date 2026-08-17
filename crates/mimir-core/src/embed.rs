//! Local ONNX embeddings via fastembed — lazy, optional, never required.
//!
//! The model downloads only on explicit request (`mimir init` /
//! `mimir embed --fetch`); a marker file gates silent loads so an agent
//! call can never block on a network fetch. Without a model everything
//! degrades to BM25-only.

use std::path::PathBuf;

use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, RerankInitOptions, RerankInitOptionsUserDefined,
    RerankerModel, TextEmbedding, TextInitOptions, TextRerank, TokenizerFiles,
    UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use rusqlite::{params, Connection, OptionalExtension};

use crate::config::Paths;
use crate::error::{Error, Result};

#[cfg(all(feature = "gpu-webgpu", feature = "gpu-cuda"))]
compile_error!(
    "gpu-webgpu and gpu-cuda are mutually exclusive: each selects a different \
     prebuilt onnxruntime binary and no binary ships with both. Pick one."
);

/// Name of the GPU backend compiled into this binary, if any.
pub fn gpu_backend() -> Option<&'static str> {
    #[cfg(feature = "gpu-webgpu")]
    {
        Some("webgpu (Vulkan/D3D12/Metal via Dawn)")
    }
    #[cfg(feature = "gpu-cuda")]
    {
        Some("cuda")
    }
    #[cfg(not(any(feature = "gpu-webgpu", feature = "gpu-cuda")))]
    {
        None
    }
}

/// Execution providers for a device setting. Empty = onnxruntime's CPU
/// default. EPs that fail to initialize at session build fall back to
/// CPU with a logged warning — "auto" never hard-fails.
fn execution_providers(device: &str) -> Vec<ort::ep::ExecutionProviderDispatch> {
    if device == "cpu" {
        return Vec::new();
    }
    let eps: Vec<ort::ep::ExecutionProviderDispatch> = vec![
        #[cfg(feature = "gpu-webgpu")]
        ort::ep::WebGPU::default().build(),
        #[cfg(feature = "gpu-cuda")]
        ort::ep::CUDA::default().build(),
    ];
    if !eps.is_empty() {
        tracing::info!(
            device,
            backend = gpu_backend(),
            "registering GPU execution provider"
        );
    }
    eps
}

pub const EMBED_BATCH: usize = 64;

/// Kinds whose body participates in semantic recall.
pub const EMBEDDABLE_KINDS: &str = "'memory', 'chunk', 'annotation', 'symbol', 'codechunk'";

/// Where a named model's weights come from. Registry models are covered by
/// fastembed's own built-in downloader; anything else (e.g. granite, wave-2
/// item #8's A/B candidate) we fetch and load ourselves via fastembed's
/// user-defined-model path — see `fetch_granite_files` / `load_granite`.
enum ModelSource {
    Registry(EmbeddingModel),
    Granite,
}

fn model_from_name(name: &str) -> Result<ModelSource> {
    Ok(match name {
        // Quantized ONNX port: 384-dim, ~34 MB — the default.
        "bge-small-en-v1.5" | "bge-small-en-v1.5-q" => {
            ModelSource::Registry(EmbeddingModel::BGESmallENV15Q)
        }
        "bge-base-en-v1.5" => ModelSource::Registry(EmbeddingModel::BGEBaseENV15),
        "all-minilm-l6-v2" => ModelSource::Registry(EmbeddingModel::AllMiniLML6V2),
        // Not in fastembed's registry (ModernBERT architecture). 384-dim,
        // same as bge-small — evaluated as a candidate replacement (wave-2
        // item #8) but NOT the default; selecting it is opt-in via
        // `mimir.toml`'s `embedding.model`.
        "granite-embedding-small-r2" => ModelSource::Granite,
        other => {
            return Err(Error::Invalid(format!(
                "unknown embedding model '{other}' (try bge-small-en-v1.5)"
            )))
        }
    })
}

const GRANITE_REPO: &str = "onnx-community/granite-embedding-small-english-r2-ONNX";
const GRANITE_ONNX_FILE: &str = "onnx/model_quantized.onnx";
const GRANITE_ONNX_DATA_FILE: &str = "onnx/model_quantized.onnx_data";
// Basename only: this is what the graph in GRANITE_ONNX_FILE references its
// external-data companion by, not the repo-relative path used to fetch it.
const GRANITE_ONNX_DATA_BASENAME: &str = "model_quantized.onnx_data";
const GRANITE_TOKENIZER_FILES: [&str; 4] = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// Fetch (or reuse already-cached) granite files into `paths.models_dir`,
/// using the same hf-hub cache convention fastembed's own registry
/// downloads use — so this coexists on disk with registry models rather
/// than adding a second cache layout. `allow_download = false` is only
/// reached here once the caller's marker-file gate already passed, mirroring
/// the registry path's semantics: a silent load still trusts the cache.
fn fetch_granite_files(
    paths: &Paths,
    allow_download: bool,
) -> Result<(PathBuf, PathBuf, [PathBuf; 4])> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(paths.models_dir.clone())
        .with_progress(allow_download)
        .build()
        .map_err(|e| Error::Embed(format!("hf-hub client: {e}")))?;
    let repo = api.model(GRANITE_REPO.to_string());
    let get = |file: &str| -> Result<PathBuf> {
        repo.get(file)
            .map_err(|e| Error::Embed(format!("downloading {GRANITE_REPO}/{file}: {e}")))
    };
    let onnx_file = get(GRANITE_ONNX_FILE)?;
    let onnx_data_file = get(GRANITE_ONNX_DATA_FILE)?;
    let tokenizer = [
        get(GRANITE_TOKENIZER_FILES[0])?,
        get(GRANITE_TOKENIZER_FILES[1])?,
        get(GRANITE_TOKENIZER_FILES[2])?,
        get(GRANITE_TOKENIZER_FILES[3])?,
    ];
    Ok((onnx_file, onnx_data_file, tokenizer))
}

/// Load granite via fastembed's user-defined-model path: unlike the
/// registry path, fastembed does no downloading/caching for us here, so
/// `fetch_granite_files` does that part first.
fn load_granite(paths: &Paths, device: &str, allow_download: bool) -> Result<TextEmbedding> {
    let (onnx_path, onnx_data_path, tok) = fetch_granite_files(paths, allow_download)?;
    let read = |p: &PathBuf| -> Result<Vec<u8>> { std::fs::read(p).map_err(|e| Error::io(p, e)) };
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read(&tok[0])?,
        config_file: read(&tok[1])?,
        special_tokens_map_file: read(&tok[2])?,
        tokenizer_config_file: read(&tok[3])?,
    };
    let model = UserDefinedEmbeddingModel::new(read(&onnx_path)?, tokenizer_files)
        // external-data ONNX: the graph file references its weights buffer
        // by this basename (onnx-community's export layout).
        .with_external_initializer(
            GRANITE_ONNX_DATA_BASENAME.to_string(),
            read(&onnx_data_path)?,
        );
    // Pooling: left at fastembed's default (Cls). Granite's own
    // 1_Pooling/config.json specifies pooling_mode_cls_token, NOT
    // config.json's unrelated classifier_pooling field — verified against
    // IBM's original sentence-transformers repo before trusting it.
    let options =
        InitOptionsUserDefined::new().with_execution_providers(execution_providers(device));
    TextEmbedding::try_new_from_user_defined(model, options)
        .map_err(|e| Error::Embed(e.to_string()))
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
    /// network on an agent's hot path. `device`: see EmbeddingConfig.
    pub fn load(paths: &Paths, name: &str, device: &str, allow_download: bool) -> Result<Embedder> {
        if !allow_download && !model_ready(paths, name) {
            return Err(Error::Invalid(format!(
                "embedding model '{name}' not downloaded; run `mimir init` or `mimir embed --fetch`"
            )));
        }
        let which = model_from_name(name)?;
        std::fs::create_dir_all(&paths.models_dir).map_err(|e| Error::io(&paths.models_dir, e))?;
        let mut model = match which {
            ModelSource::Registry(which) => {
                let options = TextInitOptions::new(which)
                    .with_cache_dir(paths.models_dir.clone())
                    .with_execution_providers(execution_providers(device))
                    .with_show_download_progress(allow_download);
                TextEmbedding::try_new(options).map_err(|e| Error::Embed(e.to_string()))?
            }
            ModelSource::Granite => load_granite(paths, device, allow_download)?,
        };
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

/// Where a named reranker's weights come from. Registry models are covered
/// by fastembed's own built-in downloader; fastembed hardcodes the fp32
/// `onnx/model.onnx` for jina-turbo, so the int8 variant of that same repo
/// (not in the registry) is fetched and loaded ourselves via fastembed's
/// user-defined-reranker path — see `fetch_jina_turbo_int8_files` /
/// `load_jina_turbo_int8`.
enum RerankerSource {
    Registry(RerankerModel),
    JinaTurboInt8,
}

fn reranker_from_name(name: &str) -> Result<RerankerSource> {
    Ok(match name {
        // ~150 MB, English, fast on CPU — the default.
        "jina-reranker-v1-turbo-en" => {
            RerankerSource::Registry(RerankerModel::JINARerankerV1TurboEn)
        }
        // Same repo/weights, int8-quantized (~38 MB): evaluated as a latency
        // candidate against the fp32 default but NOT the default itself;
        // selecting it is opt-in via `mimir.toml`'s `rerank.model`.
        "jina-reranker-v1-turbo-en-int8" => RerankerSource::JinaTurboInt8,
        "bge-reranker-base" => RerankerSource::Registry(RerankerModel::BGERerankerBase),
        "bge-reranker-v2-m3" => RerankerSource::Registry(RerankerModel::BGERerankerV2M3),
        "jina-reranker-v2-base-multilingual" => {
            RerankerSource::Registry(RerankerModel::JINARerankerV2BaseMultiligual)
        }
        other => {
            return Err(Error::Invalid(format!(
                "unknown reranker model '{other}' (try jina-reranker-v1-turbo-en)"
            )))
        }
    })
}

const JINA_TURBO_INT8_REPO: &str = "jinaai/jina-reranker-v1-turbo-en";
const JINA_TURBO_INT8_ONNX_FILE: &str = "onnx/model_int8.onnx";
// Same 4 tokenizer files fastembed's own registry path pulls for this repo
// (see fastembed's `load_tokenizer_hf_hub`) — reused verbatim here.
const JINA_TOKENIZER_FILES: [&str; 4] = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// Fetch (or reuse already-cached) int8 jina-turbo files into
/// `paths.models_dir`, same hf-hub cache convention as `fetch_granite_files`.
fn fetch_jina_turbo_int8_files(
    paths: &Paths,
    allow_download: bool,
) -> Result<(PathBuf, [PathBuf; 4])> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(paths.models_dir.clone())
        .with_progress(allow_download)
        .build()
        .map_err(|e| Error::Embed(format!("hf-hub client: {e}")))?;
    let repo = api.model(JINA_TURBO_INT8_REPO.to_string());
    let get = |file: &str| -> Result<PathBuf> {
        repo.get(file)
            .map_err(|e| Error::Embed(format!("downloading {JINA_TURBO_INT8_REPO}/{file}: {e}")))
    };
    let onnx_file = get(JINA_TURBO_INT8_ONNX_FILE)?;
    let tokenizer = [
        get(JINA_TOKENIZER_FILES[0])?,
        get(JINA_TOKENIZER_FILES[1])?,
        get(JINA_TOKENIZER_FILES[2])?,
        get(JINA_TOKENIZER_FILES[3])?,
    ];
    Ok((onnx_file, tokenizer))
}

/// Load int8 jina-turbo via fastembed's user-defined-reranker path. Unlike
/// the embedding side's ModernBERT export, this ONNX file is self-contained
/// (38 MB, well under the 2 GB protobuf external-data threshold) — no
/// `.onnx_data` companion to fetch.
fn load_jina_turbo_int8(paths: &Paths, device: &str, allow_download: bool) -> Result<TextRerank> {
    let (onnx_path, tok) = fetch_jina_turbo_int8_files(paths, allow_download)?;
    let read = |p: &PathBuf| -> Result<Vec<u8>> { std::fs::read(p).map_err(|e| Error::io(p, e)) };
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read(&tok[0])?,
        config_file: read(&tok[1])?,
        special_tokens_map_file: read(&tok[2])?,
        tokenizer_config_file: read(&tok[3])?,
    };
    let model = UserDefinedRerankingModel::new(read(&onnx_path)?, tokenizer_files);
    // RerankInitOptionsUserDefined is #[non_exhaustive] (no struct-literal
    // construction outside fastembed) and exposes no `with_execution_providers`
    // builder — its fields are still `pub`, so mutate the default in place.
    let mut options = RerankInitOptionsUserDefined::default();
    options.execution_providers = execution_providers(device);
    TextRerank::try_new_from_user_defined(model, options).map_err(|e| Error::Embed(e.to_string()))
}

/// Cross-encoder for `recall --rerank`. Same marker-file download gate
/// as `Embedder`: silent loads never touch the network.
pub struct Reranker {
    model: TextRerank,
    pub name: String,
}

impl Reranker {
    pub fn load(paths: &Paths, name: &str, device: &str, allow_download: bool) -> Result<Reranker> {
        if !allow_download && !model_ready(paths, name) {
            return Err(Error::Invalid(format!(
                "reranker model '{name}' not downloaded; run `mimir embed --fetch --rerank`"
            )));
        }
        let which = reranker_from_name(name)?;
        std::fs::create_dir_all(&paths.models_dir).map_err(|e| Error::io(&paths.models_dir, e))?;
        let model = match which {
            RerankerSource::Registry(which) => {
                let options = RerankInitOptions::new(which)
                    .with_cache_dir(paths.models_dir.clone())
                    .with_execution_providers(execution_providers(device))
                    .with_show_download_progress(allow_download);
                TextRerank::try_new(options).map_err(|e| Error::Embed(e.to_string()))?
            }
            RerankerSource::JinaTurboInt8 => load_jina_turbo_int8(paths, device, allow_download)?,
        };
        std::fs::write(marker_path(paths, name), b"ok")
            .map_err(|e| Error::io(&paths.models_dir, e))?;
        Ok(Reranker {
            model,
            name: name.to_string(),
        })
    }

    /// Score (query, doc) pairs; returns (input index, relevance score),
    /// best first.
    pub fn rerank(&mut self, query: &str, docs: Vec<String>) -> Result<Vec<(usize, f32)>> {
        let results = self
            .model
            .rerank(query.to_string(), &docs, false, None)
            .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
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
///
/// Work is committed in batches so the single SQLite writer lock is yielded
/// between chunks instead of held for the whole run, and — crucially — the
/// CPU-bound model inference happens OUTSIDE any transaction, so a big embed
/// never starves concurrent sessions. Each batch is independent: embeddings
/// carry no cross-row invariant, so a partial run just leaves the rest pending
/// for next time.
///
/// Returns the ids of every node embedded (fresh inference or dedup copy),
/// so callers can patch a live vector-matrix cache instead of dropping it.
/// WHERE clause selecting nodes that still need (re-)embedding for model ?1,
/// against `node n LEFT JOIN embedding e ON e.node_id = n.id`. Shared by
/// `embed_pending` and `has_pending` so the probe can never drift from the
/// real selection.
fn pending_where() -> String {
    format!(
        "n.deleted_at IS NULL AND n.body IS NOT NULL
           AND n.kind IN ({EMBEDDABLE_KINDS})
           AND (e.node_id IS NULL OR e.model <> ?1
                OR n.content_hash IS NULL OR e.content_hash <> n.content_hash)"
    )
}

/// Cheap existence probe for embeddable work. Callers that would otherwise
/// load the embedder speculatively (background sync, startup maintenance)
/// gate on this first — on GPU builds a model load creates a whole GPU
/// device, so a speculative load that finds nothing to do is far from free.
pub fn has_pending(conn: &Connection, model: &str) -> Result<bool> {
    let sql = format!(
        "SELECT EXISTS(SELECT 1 FROM node n
             LEFT JOIN embedding e ON e.node_id = n.id WHERE {})",
        pending_where()
    );
    Ok(conn.query_row(&sql, [model], |r| r.get(0))?)
}

pub fn embed_pending(
    conn: &Connection,
    model: &str,
    embed: &mut dyn FnMut(Vec<String>) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<i64>> {
    struct Pending {
        id: i64,
        text: String,
        hash: Vec<u8>,
    }
    let mut stmt = conn.prepare(&format!(
        "SELECT n.id, n.title, n.body, n.content_hash
         FROM node n LEFT JOIN embedding e ON e.node_id = n.id
         WHERE {}",
        pending_where()
    ))?;
    let pending: Vec<Pending> = stmt
        .query_map([model], |r| {
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
        return Ok(Vec::new());
    }
    let ids: Vec<i64> = pending.iter().map(|p| p.id).collect();

    // Nodes per write transaction. Inference is out of the lock, so the tx only
    // holds for the upserts; this bounds the hold (and memory) on a bulk import.
    const TX_BATCH: usize = 256;
    for batch in pending.chunks(TX_BATCH) {
        // Phase 1 (no write lock): decide copy-vs-embed. Reading the embedding
        // table for an identical-content vector is a plain autocommit read.
        let mut copies: Vec<(&Pending, i64, Vec<u8>)> = Vec::new();
        let mut to_embed: Vec<&Pending> = Vec::new();
        for p in batch {
            let existing: Option<(i64, Vec<u8>)> = conn
                .query_row(
                    "SELECT dim, vec FROM embedding
                     WHERE model = ?1 AND content_hash = ?2 AND node_id <> ?3 LIMIT 1",
                    params![model, p.hash, p.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            match existing {
                Some((dim, vec)) => copies.push((p, dim, vec)),
                None => to_embed.push(p),
            }
        }

        // Phase 2 (no write lock): model inference for the misses.
        let vectors = if to_embed.is_empty() {
            Vec::new()
        } else {
            embed(to_embed.iter().map(|p| p.text.clone()).collect())?
        };

        // Phase 3: take the writer lock just long enough to persist this batch.
        // IMMEDIATE so a concurrent writer waits (busy_timeout) instead of
        // erroring on a DEFERRED read→write upgrade. new_unchecked because we
        // only hold &Connection (the caller keeps the inference borrow).
        let tx =
            rusqlite::Transaction::new_unchecked(conn, rusqlite::TransactionBehavior::Immediate)?;
        for (p, dim, vec) in &copies {
            upsert(&tx, p.id, model, *dim as usize, vec, &p.hash)?;
        }
        for (p, vec) in to_embed.iter().zip(&vectors) {
            upsert(&tx, p.id, model, vec.len(), &to_blob(vec), &p.hash)?;
        }
        tx.commit()?;
    }
    Ok(ids)
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
    fn has_pending_gates_on_real_work() {
        let conn = crate::db::open_in_memory().unwrap();
        let model = "bge-small-en-v1.5";
        assert!(!has_pending(&conn, model).unwrap(), "empty store");

        let node = match crate::memory::remember(
            &conn,
            crate::memory::Remember {
                text: "SCRAM auth rejects non-ASCII passwords".into(),
                mtype: crate::model::MemoryType::Note,
                tags: vec![],
                project_id: None,
                force: false,
                actor: crate::model::Actor::System,
            },
        )
        .unwrap()
        {
            crate::memory::RememberOutcome::Created(n) => n,
            other => panic!("expected Created, got {other:?}"),
        };
        assert!(has_pending(&conn, model).unwrap(), "one unembedded memory");

        let hash: Vec<u8> = conn
            .query_row(
                "SELECT content_hash FROM node WHERE id = ?1",
                [node.id],
                |r| r.get(0),
            )
            .unwrap();
        upsert(&conn, node.id, model, 4, &to_blob(&[0.5; 4]), &hash).unwrap();
        assert!(
            !has_pending(&conn, model).unwrap(),
            "embedded at current hash"
        );
        assert!(
            has_pending(&conn, "some-other-model").unwrap(),
            "same row is still pending for a different model"
        );
    }

    #[test]
    fn embed_pending_skips_model_load_when_nothing_pending() {
        let mut m = crate::Mimir::open_in_memory().unwrap();
        assert_eq!(m.embed_pending().unwrap(), 0);
        // The probe must return before `ensure_embedder` runs — a speculative
        // load would create a GPU device on GPU builds just to do nothing.
        assert!(!m.embedder_tried, "embedder load was attempted on a no-op");
    }

    #[test]
    fn model_names() {
        assert!(model_from_name("bge-small-en-v1.5").is_ok());
        assert!(model_from_name("granite-embedding-small-r2").is_ok());
        assert!(model_from_name("nonsense").is_err());
    }

    #[test]
    fn reranker_names() {
        assert!(reranker_from_name("jina-reranker-v1-turbo-en").is_ok());
        assert!(reranker_from_name("jina-reranker-v1-turbo-en-int8").is_ok());
        assert!(reranker_from_name("nonsense").is_err());
    }

    /// Network test: exercises the actual production path (marker gate +
    /// hf-hub download/cache + user-defined-model load), not just the raw
    /// fastembed call the throwaway smoke example proved works. Ignored —
    /// run explicitly with `cargo test -p mimir-mem-core --lib -- --ignored
    /// granite_loads_via_production_path`.
    #[test]
    #[ignore]
    fn granite_loads_via_production_path() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = crate::config::Paths::under_root(tmp.path());

        let t0 = std::time::Instant::now();
        let mut embedder = Embedder::load(&paths, "granite-embedding-small-r2", "cpu", true)
            .expect("granite should load via the hf-hub download plumbing");
        eprintln!("cold load: {:?}", t0.elapsed());

        assert_eq!(embedder.dim, 384);
        assert!(model_ready(&paths, "granite-embedding-small-r2"));

        let out = embedder
            .embed(vec!["what does retry_with_backoff do".to_string()])
            .unwrap();
        let norm: f32 = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedder::embed must L2-normalize"
        );

        // Marker present now — a silent (non-downloading) reload must succeed
        // purely from the hf-hub cache, same contract as the registry path.
        let t1 = std::time::Instant::now();
        Embedder::load(&paths, "granite-embedding-small-r2", "cpu", false)
            .expect("cached granite should reload without allow_download");
        eprintln!("warm cache reload: {:?}", t1.elapsed());
    }

    /// Network test, same shape as `granite_loads_via_production_path` but
    /// for the int8 jina-turbo reranker: marker gate, hf-hub download/cache,
    /// and a user-defined-reranker load, then a sanity rerank call. Ignored —
    /// run explicitly with `cargo test -p mimir-mem-core --lib -- --ignored
    /// jina_turbo_int8_loads_via_production_path`.
    #[test]
    #[ignore]
    fn jina_turbo_int8_loads_via_production_path() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = crate::config::Paths::under_root(tmp.path());

        let t0 = std::time::Instant::now();
        let mut reranker = Reranker::load(&paths, "jina-reranker-v1-turbo-en-int8", "cpu", true)
            .expect("int8 jina-turbo should load via the hf-hub download plumbing");
        eprintln!("cold load: {:?}", t0.elapsed());
        assert!(model_ready(&paths, "jina-reranker-v1-turbo-en-int8"));

        // Obviously-relevant doc must outrank an obviously-irrelevant one.
        let results = reranker
            .rerank(
                "what does retry_with_backoff do",
                vec![
                    "retry_with_backoff retries the operation with exponential backoff \
                     between attempts"
                        .to_string(),
                    "the quarterly bake sale raised money for the school".to_string(),
                ],
            )
            .unwrap();
        assert_eq!(results[0].0, 0, "relevant doc should rank first");

        // Marker present now — a silent (non-downloading) reload must succeed
        // purely from the hf-hub cache, same contract as the registry path.
        let t1 = std::time::Instant::now();
        Reranker::load(&paths, "jina-reranker-v1-turbo-en-int8", "cpu", false)
            .expect("cached int8 jina-turbo should reload without allow_download");
        eprintln!("warm cache reload: {:?}", t1.elapsed());
    }
}
