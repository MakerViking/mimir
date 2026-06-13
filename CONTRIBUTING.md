# Contributing

Thanks for considering it. Mimir is a small, focused Rust workspace and
contributions are welcome — bug reports, language adapters, and docs
especially.

## Ground rules

- **Open an issue first** for anything non-trivial, so we agree on the shape
  before you write code.
- **Keep the invariants.** The design has a few deliberate, load-bearing
  choices: one SQLite file, everything-is-a-node, exact brute-force vector
  scan (no ANN below ~200k vectors), `mimir-core` stays sync (no tokio),
  zero telemetry. If a change touches one of these, call it out.
- **Security issues go through [SECURITY.md](SECURITY.md)**, not public PRs.

## Development

```sh
cargo test --workspace                                   # fast unit + e2e
cargo clippy --workspace --all-targets -- -D warnings    # must be clean
cargo fmt --all                                          # must be applied
cargo test --release scaling_profile -- --ignored --nocapture  # scale profile
```

CI runs fmt + clippy + tests on Linux, macOS, and Windows; all three must be
green. Please add a test with any behavior change.

## Adding a code-graph language

Each language is a self-contained adapter in
`crates/mimir-graph/src/languages.rs`: add the `Lang` variant, the file
extensions, the tree-sitter grammar, and the `definition`/`call`/`imports`
matchers, then a round-trip test in `crates/mimir-graph/src/lib.rs`. Copy an
existing language (Go is a compact example) and adjust the node-kind names
for the new grammar.

## License

By contributing you agree your work is licensed under the project's dual
MIT / Apache-2.0 license.
