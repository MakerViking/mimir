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
    if crate::config::EmbeddingDevice::resolve(device) == crate::config::EmbeddingDevice::Cpu {
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

/// How long a model that failed to load is left alone before a caller may
/// try again. Loading is the expensive step (an ONNX session; on GPU builds
/// a whole device, and creating and dropping those repeatedly was measured
/// to leak driver memory), so a broken model must not be retried on every
/// recall, nor every few seconds for the life of a daemon. Five minutes
/// still lets a long-lived server recover by itself once whatever was wrong
/// (a half-written cache, a busy GPU) is fixed.
pub(crate) const LOAD_RETRY_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(300);

/// The longest a failed model is left alone. Each failure in a row doubles
/// the wait up to this: a GPU that was merely busy is retried within
/// minutes, while a load that keeps failing settles at one attempt an hour
/// instead of one every five minutes for the life of a daemon, each of
/// which costs the device set-up described above.
pub(crate) const LOAD_RETRY_MAX: std::time::Duration = std::time::Duration::from_secs(3600);

/// How long to leave a model alone after its `failures`-th failure in a row.
pub(crate) fn retry_after(failures: u32) -> std::time::Duration {
    let doublings = failures.saturating_sub(1).min(16);
    LOAD_RETRY_COOLDOWN
        .saturating_mul(1u32 << doublings)
        .min(LOAD_RETRY_MAX)
}

/// One lazily loaded model, and what this process knows about it.
///
/// ```text
///   Unloaded ── load succeeds ─▶ Loaded        (until the process exits)
///   Unloaded ── load fails ────▶ Failed
///   Failed ── `retry_after(failures)` has passed, or the caller now
///             allows the download the failed attempt was refused
///          ── load ────────────▶ Loaded | Failed (failures + 1)
/// ```
///
/// A failure is remembered so a hot path does not pay for the same doomed
/// load on every call, but it is not final: a server that started while
/// the weights were unusable must not stay BM25-only until someone
/// restarts it, and an explicit `--fetch` must be able to do what a silent
/// load could not.
pub(crate) enum ModelSlot<T> {
    /// No load has been attempted.
    Unloaded,
    Loaded(T),
    Failed {
        at: std::time::Instant,
        /// Whether the failed attempt was allowed to download the weights.
        /// One that was not says nothing about what a download would do.
        allowed_download: bool,
        /// Failed loads in a row, this one included. Sets the wait before
        /// the next attempt (see [`retry_after`]).
        failures: u32,
    },
}

impl<T> ModelSlot<T> {
    pub(crate) fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            ModelSlot::Loaded(model) => Some(model),
            ModelSlot::Unloaded | ModelSlot::Failed { .. } => None,
        }
    }

    pub(crate) fn is_loaded(&self) -> bool {
        matches!(self, ModelSlot::Loaded(_))
    }

    /// Whether a caller arriving at `now` should attempt a load.
    pub(crate) fn wants_load(&self, allow_download: bool, now: std::time::Instant) -> bool {
        match self {
            ModelSlot::Unloaded => true,
            ModelSlot::Loaded(_) => false,
            ModelSlot::Failed {
                at,
                allowed_download,
                failures,
            } => {
                (allow_download && !allowed_download)
                    || now.saturating_duration_since(*at) >= retry_after(*failures)
            }
        }
    }

    /// Record the result of a load attempted at `now`, handing back the
    /// error of a failed one for the caller to log.
    pub(crate) fn settle(
        &mut self,
        loaded: Result<T>,
        allow_download: bool,
        now: std::time::Instant,
    ) -> Option<Error> {
        match loaded {
            Ok(model) => {
                *self = ModelSlot::Loaded(model);
                None
            }
            Err(err) => {
                let failures = match self {
                    ModelSlot::Failed { failures, .. } => failures.saturating_add(1),
                    ModelSlot::Unloaded | ModelSlot::Loaded(_) => 1,
                };
                *self = ModelSlot::Failed {
                    at: now,
                    allowed_download: allow_download,
                    failures,
                };
                Some(err)
            }
        }
    }
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
    // `as_chunks::<4>` rather than `chunks_exact(4)`: the const generic hands
    // back `[u8; 4]` directly, so `from_le_bytes` takes it without the manual
    // re-indexing that could silently transpose a byte. A trailing partial
    // chunk is discarded either way — a blob whose length is not a multiple of
    // 4 is not a vector this wrote.
    //
    // Also what `clippy::chunks_exact_to_as_chunks` asks for. That lint is
    // newer than some toolchains in use here, so this is written to satisfy it
    // rather than `#[allow]`ed: an allow of a lint an older clippy has never
    // heard of trips `unknown_lints`, which fails the same `-D warnings` gate
    // from the other side.
    let (chunks, _) = blob.as_chunks::<4>();
    chunks.iter().copied().map(f32::from_le_bytes).collect()
}

/// Which nodes can be embedded at all, as a predicate on `node n`: live, with
/// a body, of a kind whose text takes part in semantic recall. Every count
/// of "embeddable" in the crate reads this, so none can drift from what
/// [`embed_pending`] selects.
pub(crate) fn embeddable_where() -> String {
    format!("n.deleted_at IS NULL AND n.body IS NOT NULL AND n.kind IN ({EMBEDDABLE_KINDS})")
}

/// Whether an embeddable node's embedding row is missing or no longer
/// describes it for model ?1, as a predicate on
/// `node n LEFT JOIN embedding e ON e.node_id = n.id`: no row, a row from
/// another model, or a row made from different content. A node with no
/// `content_hash` of its own counts too — there is nothing to compare the
/// row against — and [`embed_pending`] gives it one, so that state lasts
/// for one pass.
const STALE_WHERE: &str = "(e.node_id IS NULL OR e.model <> ?1
                OR n.content_hash IS NULL OR e.content_hash <> n.content_hash)";

/// WHERE clause selecting nodes that still need (re-)embedding for model ?1,
/// against `node n LEFT JOIN embedding e ON e.node_id = n.id`. Shared by
/// `embed_pending`, `has_pending` and `coverage` so neither the probe nor
/// the reported number can drift from the real selection.
fn pending_where() -> String {
    format!("{} AND {STALE_WHERE}", embeddable_where())
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

/// `(embedded, embeddable)` for `model`: how many embeddable nodes there
/// are, and how many of them [`embed_pending`] would leave alone.
///
/// Counted from the pending predicate rather than from the `embedding`
/// table, because a row there is not the same thing as coverage: it may
/// belong to a deleted node, to another model, or to content that has
/// since changed. `embedded` can never exceed `embeddable`.
pub fn coverage(conn: &Connection, model: &str) -> Result<(i64, i64)> {
    let sql = format!(
        "SELECT count(*), COALESCE(sum(CASE WHEN {STALE_WHERE} THEN 1 ELSE 0 END), 0)
         FROM node n LEFT JOIN embedding e ON e.node_id = n.id
         WHERE {}",
        embeddable_where()
    );
    let (embeddable, pending): (i64, i64) =
        conn.query_row(&sql, [model], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok((embeddable - pending, embeddable))
}

/// The `content_hash` a node of `kind` carries for `body`: the normalized
/// dedup key `memory` writes for a memory, the plain body hash the indexer
/// writes for everything else. Used for the nodes that reached the store
/// without one.
fn content_hash_for(kind: &str, body: &str) -> Vec<u8> {
    if kind == crate::model::Kind::Memory.as_str() {
        crate::memory::content_hash(body)
    } else {
        blake3::hash(body.as_bytes()).as_bytes().to_vec()
    }
}

/// Embed every node whose content is new or changed for this model.
/// Unchanged content never reaches the model: embeddings are keyed by
/// content hash, and identical text elsewhere in the store is copied.
///
/// A node stored without a `content_hash` (a memory synced from a peer that
/// sent none, a docs note written by an older build) is given one here, in
/// the same transaction as its embedding row. Without that the node would
/// stay pending for ever — see [`STALE_WHERE`] — and be re-embedded on
/// every pass. `updated_at` is left alone: nothing about the node changed,
/// and bumping it would push the row back out over sync.
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
pub fn embed_pending(
    conn: &Connection,
    model: &str,
    embed: &mut dyn FnMut(Vec<String>) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<i64>> {
    struct Pending {
        id: i64,
        text: String,
        hash: Vec<u8>,
        /// Set when the node itself has no hash yet: the body `hash` was
        /// computed from, so the hash is written back only if the row still
        /// holds that body.
        hashed_body: Option<String>,
    }
    let mut stmt = conn.prepare(&format!(
        "SELECT n.id, n.title, n.body, n.content_hash, n.kind
         FROM node n LEFT JOIN embedding e ON e.node_id = n.id
         WHERE {}",
        pending_where()
    ))?;
    let pending: Vec<Pending> = stmt
        .query_map([model], |r| {
            let title: Option<String> = r.get(1)?;
            let body: String = r.get(2)?;
            let hash: Option<Vec<u8>> = r.get(3)?;
            let kind: String = r.get(4)?;
            let text = match &title {
                Some(t) if !t.is_empty() => format!("{t}\n{body}"),
                _ => body.clone(),
            };
            let (hash, hashed_body) = match hash {
                Some(hash) => (hash, None),
                None => (content_hash_for(&kind, &body), Some(body)),
            };
            Ok(Pending {
                id: r.get(0)?,
                hash,
                hashed_body,
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
        // Only the rows whose embedding was just written: a node the model
        // returned no vector for stays pending and is retried.
        let written = copies
            .iter()
            .map(|(p, _, _)| *p)
            .chain(to_embed.iter().zip(&vectors).map(|(p, _)| *p));
        // Inference ran outside any lock, so the row may have been
        // rewritten since it was read (a sync apply from a peer that sends
        // no hash is exactly the population this backfill serves). The hash
        // goes on the row only while the row still holds the body it was
        // computed from; otherwise the row stays without one, and pending.
        for p in written {
            if let Some(body) = &p.hashed_body {
                tx.execute(
                    "UPDATE node SET content_hash = ?2
                     WHERE id = ?1 AND content_hash IS NULL AND body = ?3",
                    params![p.id, p.hash, body],
                )?;
            }
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

    /// `from_blob` decodes every stored embedding, so its edge behaviour is
    /// worth pinning rather than assuming: a trailing partial chunk is
    /// dropped, not padded into a bogus float. A blob whose length is not a
    /// multiple of 4 was not written by `to_blob`, and inventing a final
    /// value from 1-3 stray bytes would put a garbage dimension into a
    /// vector that then scores against every query.
    #[test]
    fn from_blob_drops_a_trailing_partial_float() {
        let v = vec![1.0f32, -2.0];
        let mut truncated = to_blob(&v);
        truncated.extend_from_slice(&[0xAB, 0xCD, 0xEF]); // 3 stray bytes
        assert_eq!(from_blob(&truncated), v);

        assert!(from_blob(&[]).is_empty());
        assert!(from_blob(&[1, 2, 3]).is_empty(), "no complete float at all");
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

    /// A canned model: a fixed vector per text, counting how many texts it
    /// was actually asked to embed.
    fn counting_embedder(
        calls: &std::cell::Cell<usize>,
    ) -> impl FnMut(Vec<String>) -> Result<Vec<Vec<f32>>> + '_ {
        move |texts| {
            calls.set(calls.get() + texts.len());
            Ok(texts.iter().map(|_| vec![0.5; 4]).collect())
        }
    }

    /// Inference runs outside any lock. A row rewritten while its old body
    /// is being embedded must not be stamped with the hash of that old
    /// body: the row would then look embedded, with a vector and a hash for
    /// text it no longer holds, and nothing would embed the new text.
    #[test]
    fn a_body_rewritten_during_inference_stays_pending() {
        use crate::model::{Kind, NewNode};
        let conn = crate::db::open_in_memory().unwrap();
        let model = "bge-small-en-v1.5";
        let mut memory = NewNode::new(Kind::Memory);
        memory.title = Some("Deploys go to eu-north-1".into());
        memory.body = Some("Deploys go to eu-north-1".into());
        let memory = crate::store::insert_node(&conn, memory).unwrap();
        assert!(memory.content_hash.is_none());

        let mut rewriting = |texts: Vec<String>| -> Result<Vec<Vec<f32>>> {
            // What a sync apply from a peer that sends no hash does.
            conn.execute(
                "UPDATE node SET body = 'Deploys go to eu-west-1' WHERE id = ?1",
                [memory.id],
            )
            .unwrap();
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        };
        embed_pending(&conn, model, &mut rewriting).unwrap();

        let after = crate::store::get_node(&conn, memory.id).unwrap();
        assert_eq!(after.content_hash, None, "hash of a body it no longer has");
        assert!(
            has_pending(&conn, model).unwrap(),
            "the rewritten text still has to be embedded"
        );
    }

    /// A node stored without a `content_hash` used to be pending for ever:
    /// the embedding row got a hash, the node never did, and "the node has no
    /// hash" is itself a reason to be pending. Every pass re-embedded it and
    /// `has_pending` never went false, which also defeated the probe that
    /// keeps a no-op pass from loading the model.
    #[test]
    fn a_node_stored_without_a_hash_is_embedded_once() {
        use crate::model::{Kind, NewNode};
        let conn = crate::db::open_in_memory().unwrap();
        let model = "bge-small-en-v1.5";

        // A memory as a sync apply stores it when the wire carried no hash,
        // and a docs note as older builds wrote it.
        let mut memory = NewNode::new(Kind::Memory);
        memory.title = Some("Deploys go to eu-north-1".into());
        memory.body = Some("Deploys go to eu-north-1".into());
        let memory = crate::store::insert_node(&conn, memory).unwrap();
        let mut note = NewNode::new(Kind::Annotation);
        note.title = Some("Runbooks".into());
        note.body = Some("Runbooks for the on-call rotation".into());
        let note = crate::store::insert_node(&conn, note).unwrap();
        assert!(memory.content_hash.is_none() && note.content_hash.is_none());
        conn.execute("UPDATE node SET updated_at = 1000", [])
            .unwrap();

        let calls = std::cell::Cell::new(0);
        let first = embed_pending(&conn, model, &mut counting_embedder(&calls)).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(calls.get(), 2);
        assert!(
            !has_pending(&conn, model).unwrap(),
            "one pass must leave nothing pending"
        );

        let second = embed_pending(&conn, model, &mut counting_embedder(&calls)).unwrap();
        assert!(
            second.is_empty(),
            "a second pass embeds nothing: {second:?}"
        );
        assert_eq!(calls.get(), 2, "and never reaches the model");

        // The node now carries the hash its kind is keyed by everywhere else,
        // so later dedup and change detection see it; it was not edited.
        let memory = crate::store::get_node(&conn, memory.id).unwrap();
        let note = crate::store::get_node(&conn, note.id).unwrap();
        assert_eq!(
            memory.content_hash,
            Some(crate::memory::content_hash("Deploys go to eu-north-1"))
        );
        assert_eq!(
            note.content_hash,
            Some(
                blake3::hash(b"Runbooks for the on-call rotation")
                    .as_bytes()
                    .to_vec()
            )
        );
        assert_eq!((memory.updated_at, note.updated_at), (1000, 1000));
    }

    /// `coverage` is the pending predicate counted the other way, so the
    /// number a dashboard shows can never disagree with what `embed_pending`
    /// would do next.
    #[test]
    fn coverage_counts_what_embed_pending_would_leave_alone() {
        use crate::model::{Kind, NewNode};
        let conn = crate::db::open_in_memory().unwrap();
        let model = "bge-small-en-v1.5";
        assert_eq!(coverage(&conn, model).unwrap(), (0, 0));

        let add = |kind: Kind, body: Option<&str>| {
            let mut new = NewNode::new(kind);
            new.title = Some("t".into());
            new.body = body.map(str::to_string);
            new.content_hash = body.map(|b| blake3::hash(b.as_bytes()).as_bytes().to_vec());
            crate::store::insert_node(&conn, new).unwrap()
        };
        let current = add(Kind::Memory, Some("embedded at the current hash"));
        let stale = add(Kind::Chunk, Some("edited since it was embedded"));
        let other_model = add(Kind::Symbol, Some("embedded by a retired model"));
        let _missing = add(Kind::CodeChunk, Some("never embedded"));
        let gone = add(Kind::Memory, Some("embedded, then deleted"));
        let _bodiless = add(Kind::Symbol, None);
        let _not_embeddable = add(Kind::File, Some("files are not embedded"));

        let hash_of = |n: &crate::model::Node| n.content_hash.clone().unwrap();
        upsert(
            &conn,
            current.id,
            model,
            4,
            &to_blob(&[0.5; 4]),
            &hash_of(&current),
        )
        .unwrap();
        upsert(
            &conn,
            stale.id,
            model,
            4,
            &to_blob(&[0.5; 4]),
            b"an older hash",
        )
        .unwrap();
        upsert(
            &conn,
            other_model.id,
            "retired-model",
            4,
            &to_blob(&[0.5; 4]),
            &hash_of(&other_model),
        )
        .unwrap();
        upsert(
            &conn,
            gone.id,
            model,
            4,
            &to_blob(&[0.5; 4]),
            &hash_of(&gone),
        )
        .unwrap();
        crate::store::soft_delete(&conn, gone.id, crate::model::Actor::System, None).unwrap();

        // Four embedding rows exist; one describes a live node as it is now.
        let rows: i64 = conn
            .query_row("SELECT count(*) FROM embedding", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 4);
        assert_eq!(coverage(&conn, model).unwrap(), (1, 4));

        let calls = std::cell::Cell::new(0);
        let embedded = embed_pending(&conn, model, &mut counting_embedder(&calls)).unwrap();
        assert_eq!(embedded.len(), 3, "exactly the gap coverage reported");
        assert_eq!(coverage(&conn, model).unwrap(), (4, 4));
    }

    #[test]
    fn embed_pending_skips_model_load_when_nothing_pending() {
        let mut m = crate::Mimir::open_in_memory().unwrap();
        assert_eq!(m.embed_pending().unwrap(), 0);
        // The probe must return before `ensure_embedder` runs — a speculative
        // load would create a GPU device on GPU builds just to do nothing.
        assert!(
            matches!(m.embedder, ModelSlot::Unloaded),
            "embedder load was attempted on a no-op"
        );
    }

    /// The slot's whole policy, on a clock the test controls.
    #[test]
    fn a_failed_load_is_retried_after_the_cooldown_or_when_a_download_is_newly_allowed() {
        use std::time::{Duration, Instant};
        let t0 = Instant::now();
        let just_before = t0 + LOAD_RETRY_COOLDOWN - Duration::from_secs(1);
        let after = t0 + LOAD_RETRY_COOLDOWN;
        let fail = || Err::<u8, _>(Error::Embed("no weights".into()));

        let mut slot: ModelSlot<u8> = ModelSlot::Unloaded;
        assert!(slot.wants_load(false, t0), "nothing was tried yet");
        assert!(slot.get_mut().is_none() && !slot.is_loaded());

        // A silent load failed.
        assert!(slot.settle(fail(), false, t0).is_some());
        assert!(
            !slot.wants_load(false, just_before),
            "not on every call: the cooldown holds"
        );
        assert!(
            slot.wants_load(true, just_before),
            "a caller who allows the download is not bound by a failure that could not download"
        );
        assert!(slot.wants_load(false, after), "the cooldown has passed");

        // A load that was allowed to download failed too. It is the second
        // failure in a row, so the wait before the next attempt doubles.
        assert!(slot.settle(fail(), true, t0).is_some());
        assert!(
            !slot.wants_load(true, just_before),
            "allowing a download again is no new information"
        );
        assert!(
            !slot.wants_load(true, after),
            "a second failure waits longer"
        );
        let after_two = t0 + retry_after(2);
        assert_eq!(retry_after(2), LOAD_RETRY_COOLDOWN * 2);
        assert!(slot.wants_load(true, after_two) && slot.wants_load(false, after_two));
        // The wait stops growing at the cap, however long it keeps failing.
        assert_eq!(retry_after(1), LOAD_RETRY_COOLDOWN);
        assert_eq!(retry_after(40), LOAD_RETRY_MAX);

        // Loaded is the end of it.
        assert!(slot.settle(Ok(7), false, after).is_none());
        assert!(slot.is_loaded());
        assert_eq!(slot.get_mut().copied(), Some(7));
        assert!(!slot.wants_load(true, after + LOAD_RETRY_COOLDOWN));
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
