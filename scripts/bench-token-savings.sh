#!/usr/bin/env bash
# Mimir token-savings benchmark: RTK vs Mimir filters (identical input via fake
# binaries on PATH), plus the Mimir-only features (outline/peek) and a coverage
# breadth comparison. All token counts use ONE tokenizer (`mimir tokens`).
set -u
REPO=/home/thomash/Koding/projects/Mimir
M=$REPO/target/debug/mimir
export MIMIR_HOME=/tmp/mimir-bench/home   # isolate: don't pollute the real store
B=/tmp/mimir-bench
FX=$B/fixtures
FAKE=$B/fakebin
REPORT=$B/REPORT.md
mkdir -p "$FX" "$FAKE" "$MIMIR_HOME"

tok() { "$M" tokens; }                     # stdin -> token count
pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if(a==0){print "0"} else {printf "%.0f", 100*(a-b)/a} }'; }

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# cargo: REAL output from this workspace (force a small rebuild).
( cd "$REPO" && cargo clean -p mimir-mem-proxy >/dev/null 2>&1 && cargo build 2>&1 ) > "$FX/cargo.txt"
# git: REAL clone-with-progress output (the noisy case git filters target).
rm -rf "$B/cloned"
git clone --no-local --progress "file://$REPO/.git" "$B/cloned" > "$FX/git.txt" 2>&1
rm -rf "$B/cloned"
# npm: synthetic but realistic (no node project on this box) — LABELLED synthetic.
cat > "$FX/npm.txt" <<'EOF'
npm WARN deprecated har-validator@5.1.5: this library is no longer supported
npm WARN deprecated uuid@3.4.0: Please upgrade to uuid@7 or higher
npm WARN deprecated request@2.88.2: request has been deprecated
npm WARN deprecated core-js@2.6.12: core-js@<3.23.3 is no longer maintained
npm WARN deprecated rollup-plugin-terser@7.0.2: This package has been deprecated
npm notice
npm notice New major version of npm available! 9.8.1 -> 10.5.0
npm notice
added 1423 packages, and audited 1424 packages in 18s
243 packages are looking for funding
  run `npm fund` for details
found 3 high severity vulnerabilities
  run `npm audit fix` to address them
EOF
# pytest: synthetic but realistic — LABELLED synthetic.
{
  echo "============================= test session starts =============================="
  echo "platform linux -- Python 3.12.3, pytest-8.2.0, pluggy-1.5.0"
  echo "rootdir: /home/user/project"
  echo "plugins: cov-5.0.0, anyio-4.3.0, hypothesis-6.100.1"
  echo "collected 248 items"
  echo
  for f in a b c d e f g h i j k l; do
    echo "tests/test_$f.py ........................................                 [ $((RANDOM%99))%]"
  done
  echo "tests/test_z.py ....................F...                                    [100%]"
  echo
  echo "=================================== FAILURES ==================================="
  echo "______________________________ test_z_edgecase ________________________________"
  echo "    assert result == 42"
  echo "E   assert 41 == 42"
  echo "=========================== 1 failed, 247 passed ==============================="
} > "$FX/pytest.txt"
# docker: synthetic — a command Mimir has NO specific handler for (generic only).
{
  for i in $(seq 1 20); do echo "#$i [internal] load build context"; echo " => => transferring context: $((i*137))kB"; done
  echo " => [builder 1/6] FROM docker.io/library/node:20"
  echo " => => resolve docker.io/library/node:20"
  echo "Successfully built a1b2c3d4e5f6"
  echo "Successfully tagged myapp:latest"
} > "$FX/docker.txt"

# Fake binaries that emit the fixtures (identical input for both tools).
for name in cargo git npm pytest docker; do
  cat > "$FAKE/$name" <<EOF
#!/usr/bin/env bash
cat "$FX/$name.txt"
exit 0
EOF
  chmod +x "$FAKE/$name"
done

# ---------------------------------------------------------------------------
# 1. Filters: RTK vs Mimir on identical input
# ---------------------------------------------------------------------------
{
echo "# Mimir token-savings benchmark"
echo
echo "_Tokenizer: \`mimir tokens\` (tiktoken o200k_base) for every measurement._"
echo "_cargo + git fixtures are REAL output from this machine; npm/pytest/docker are realistic synthetic fixtures._"
echo
echo "## 1. Command-output filtering — RTK vs Mimir (identical input)"
echo
printf '| command | raw tok | RTK tok | RTK saved | Mimir tok | Mimir saved |\n'
printf '|---|---:|---:|---:|---:|---:|\n'
} > "$REPORT"

# Real per-tool subcommands so each tool dispatches its correct handler.
declare -A SUB=( [cargo]="build" [git]="clone url" [npm]="install" [pytest]="" [docker]="build ." )
for name in cargo git npm pytest docker; do
  sub=${SUB[$name]}
  raw=$(cat "$FX/$name.txt" | tok)
  rtk_out=$(PATH="$FAKE:$PATH" rtk $name $sub 2>&1); rtk_t=$(printf '%s' "$rtk_out" | tok)
  mim_out=$(PATH="$FAKE:$PATH" "$M" run -- $name $sub 2>&1); mim_t=$(printf '%s' "$mim_out" | tok)
  printf '| %s | %s | %s | %s%% | %s | %s%% |\n' \
    "$name $sub" "$raw" "$rtk_t" "$(pct "$raw" "$rtk_t")" "$mim_t" "$(pct "$raw" "$mim_t")" >> "$REPORT"
done

# ---------------------------------------------------------------------------
# 2. outline vs reading whole files (Mimir-only; RTK has no equivalent)
# ---------------------------------------------------------------------------
{
echo
echo "## 2. \`mimir outline\` vs reading whole files (no RTK equivalent)"
echo
printf '| file | full tok | outline tok | saved |\n'
printf '|---|---:|---:|---:|\n'
} >> "$REPORT"
tot_full=0; tot_out=0; nfiles=0
while IFS= read -r f; do
  full=$(cat "$f" | tok)
  out=$(cd "$REPO" && "$M" outline "$f" 2>/dev/null | tok)
  tot_full=$((tot_full+full)); tot_out=$((tot_out+out)); nfiles=$((nfiles+1))
  rel=${f#$REPO/}
  # only print the biggest few to keep the table short
  echo "$full|$out|$rel"
done < <(find "$REPO/crates" -name '*.rs' | sort) > "$B/outline_rows.txt"
# top 6 files by full tokens
sort -t'|' -k1 -rn "$B/outline_rows.txt" | head -6 | while IFS='|' read -r full out rel; do
  printf '| %s | %s | %s | %s%% |\n' "$rel" "$full" "$out" "$(pct "$full" "$out")" >> "$REPORT"
done
printf '| **TOTAL (%s files)** | **%s** | **%s** | **%s%%** |\n' \
  "$nfiles" "$tot_full" "$tot_out" "$(pct "$tot_full" "$tot_out")" >> "$REPORT"

# ---------------------------------------------------------------------------
# 3. peek vs reading whole file (Mimir-only)
# ---------------------------------------------------------------------------
( cd "$REPO" && "$M" graph build >/dev/null 2>&1 )
{
echo
echo "## 3. \`mimir peek <symbol>\` vs reading the whole file (no RTK equivalent)"
echo
printf '| symbol | file tok | peek tok | saved |\n'
printf '|---|---:|---:|---:|\n'
} >> "$REPORT"
for sym in optimize_request filter_output record count add_cache_breakpoint; do
  pj=$(cd "$REPO" && "$M" --json peek "$sym" 2>/dev/null)
  [ -z "$pj" ] && continue
  path=$(printf '%s' "$pj" | jq -r '.path // empty' 2>/dev/null)
  ptok=$(printf '%s' "$pj" | jq -r '.tokens_after // empty' 2>/dev/null)
  ftok=$(printf '%s' "$pj" | jq -r '.tokens_before // empty' 2>/dev/null)
  [ -z "$ftok" ] && continue
  printf '| %s (`%s`) | %s | %s | %s%% |\n' "$sym" "$path" "$ftok" "$ptok" "$(pct "$ftok" "$ptok")" >> "$REPORT"
done

# ---------------------------------------------------------------------------
# 4. Coverage breadth: which commands each tool rewrites
# ---------------------------------------------------------------------------
{
echo
echo "## 4. Coverage breadth — which commands each tool wraps (\`rewrite\`)"
echo
printf '| command | RTK | Mimir |\n'
printf '|---|:--:|:--:|\n'
} >> "$REPORT"
rtk_n=0; mim_n=0
for cmd in "cargo build" "git status" "git clone x" "npm install" "pnpm i" "yarn" "pytest" "go build" "make" "docker build ." "kubectl get pods" "terraform plan" "ls -la" "grep -r x" "rg foo" "find . -name x" "pip install x" "gradle build" "mvn test" "jest" "eslint ." "tsc" "df -h" "ps aux"; do
  r=$(rtk rewrite "$cmd" 2>/dev/null); [ -n "$r" ] && [ "$r" != "$cmd" ] && rk="✓" && rtk_n=$((rtk_n+1)) || rk="·"
  m=$("$M" rewrite "$cmd" 2>/dev/null); mq=$?; [ $mq -eq 0 ] && [ -n "$m" ] && mk="✓" && mim_n=$((mim_n+1)) || mk="·"
  printf '| `%s` | %s | %s |\n' "$cmd" "$rk" "$mk" >> "$REPORT"
done
printf '| **TOTAL wrapped** | **%s / 24** | **%s / 24** |\n' "$rtk_n" "$mim_n" >> "$REPORT"

echo >> "$REPORT"
cat "$REPORT"
