# Mimir sync hub — `mimir serve`.
#
# The hub is a normal Mimir store reachable over HTTP; it performs no
# embedding, so the runtime image needs only the static binary + glibc.
# Build & run with docker-compose.yml, or:
#   docker build -t mimir-hub .
#   docker run -e MIMIR_SYNC_TOKEN=... -v mimir:/data -p 7777:7777 mimir-hub

# trixie (glibc 2.41), not bookworm (2.36): the prebuilt ONNX Runtime that
# `ort` (via `fastembed`) downloads is linked against glibc >=2.38 and
# references __isoc23_strtoll, so it won't link on bookworm.
FROM rust:1-trixie AS builder
WORKDIR /src
COPY . .
RUN cargo build --release -p mimir-mem

# Runtime base must match the builder's glibc (>=2.38) for the static ORT code.
FROM debian:trixie-slim
RUN useradd -m -u 10001 mimir && mkdir -p /data && chown mimir /data
COPY --from=builder /src/target/release/mimir /usr/local/bin/mimir
USER mimir
ENV MIMIR_HOME=/data
VOLUME /data
EXPOSE 7777
# Binds all interfaces inside the container; publish/proxy it as you see fit
# (tailnet or TLS reverse proxy — see docs/sync.md).
ENTRYPOINT ["mimir", "serve", "--bind", "0.0.0.0:7777"]
