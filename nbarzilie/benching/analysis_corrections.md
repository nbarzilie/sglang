# Analysis Corrections for the PD Benchmark JSONL Flow

## Purpose

The current analysis workflow described in `nbarzilie/benching/analysis.md`
was built for raw text logs. It expects files under `input/`, cleans text into
`cleaned_input/`, converts cleaned blocks into `csv/benchmark_results.csv`, and
plots mean TTFT and mean ITL into `result_png/`.

The new benchmark flow writes structured results directly from
`sglang.bench_serving --output-file` into:

```text
/logs/pd_bench/<run_id>/<backend>/<case>/<dataset>/results/c<concurrency>_r<rep>.jsonl
```

That JSONL must become the primary source of truth. Text logs should remain
debug evidence only. The analysis flow should be updated so it extracts all
usable signal from the JSONL records, merges run metadata, checks data quality,
summarizes repeated runs, compares NIXL and Mooncake pairwise, computes
SLO-constrained throughput, and produces plots and a report.

## Current Flow Limitations

The existing flow is useful for older logs, but it leaves too much information
unused for the new benchmark output:

- It parses human-readable `.log` text even though `bench_serving` now emits
  structured JSONL.
- It relies on filename metadata instead of the richer directory layout and
  `run_meta.json`.
- It focuses on `mean_ttft_ms` and `mean_itl_ms`, while the JSONL contains p99
  TTFT, TPOT, ITL, throughput, request counts, token counts, concurrency, and
  error details.
- It uses simple mean aggregation and does not report median, standard
  deviation, min, max, coefficient of variation, or missing-repetition status.
- It does not compute pairwise backend speedup ratios.
- It does not compute the highest throughput that satisfies latency SLOs.
- It does not validate fingerprints, cache mode, output-details mode, image,
  git SHA, GPU layout, or model consistency.
- It cannot cleanly combine Phase 1 shard A, Phase 1 shard B, cold radix runs,
  and future Phase 2 runs into one coherent report.

Keep the old log cleaner as a legacy fallback, but new analysis should never
depend on text-log parsing when JSONL exists.

## Target Flow

The new primary pipeline should be:

```text
/logs/pd_bench/<run_id>/**/results/*.jsonl
  -> csv/benchmark_results_jsonl.csv
  -> csv/benchmark_summary.csv
  -> csv/backend_pairwise.csv
  -> csv/slo_summary.csv
  -> csv/data_quality.csv
  -> result_png/*.png
  -> report.md
```

Recommended scripts:

```text
nbarzilie/benching/scripts/jsonl_to_csv.py
nbarzilie/benching/scripts/summarize_jsonl_results.py
nbarzilie/benching/scripts/plot_jsonl_comparison.py
nbarzilie/benching/scripts/build_jsonl_report.py
```

A single script can implement all stages initially, but the logical outputs
above should remain separate. The important correction is the data model, not
the exact file split.

## Input Discovery

The parser should accept one or more run roots:

```bash
python3 nbarzilie/benching/scripts/analyze_jsonl.py \
  --run-root /logs/pd_bench/llama_phase1_a \
  --run-root /logs/pd_bench/llama_phase1_b \
  --out-dir nbarzilie/benching/jsonl_analysis
```

It should recursively discover:

```text
<run_root>/<backend>/<case>/<dataset>/results/c<concurrency>_r<rep>.jsonl
```

It should support all current backends and datasets:

```text
backend: nixl, mooncake
dataset: rand, sharegpt, radixcache
case examples:
  Ptp2_Dtp2_Pdp1_Ddp1
  Ptp4_Dtp4_Pdp1_Ddp1
  Ptp2_Dtp4_Pdp1_Ddp1
  Ptp4_Dtp2_Pdp1_Ddp1
```

Do not assume only one run ID is being analyzed. Phase 1 is intentionally split
across shards. The analysis must combine compatible shards while preserving
the original `run_id` in every row.

## JSONL Record Selection

Each `c<concurrency>_r<rep>.jsonl` file normally contains one record. Because
`bench_serving` appends to existing files, a file can contain more than one
record if a run was repeated without deleting the output file.

Required behavior:

- Parse every complete JSON line.
- Ignore blank lines.
- If a file has one record, use it.
- If a file has multiple records, use the last complete record by default and
  emit a warning into `data_quality.csv`.
- Add an option such as `--multi-record-mode all|last|fail`.
- Default to `last` for practical recovery.
- Use `fail` for strict publication-quality analysis.

The selected record should retain:

```text
jsonl_path
jsonl_record_index
jsonl_record_count
```

This makes rerun/appended files auditable.

## Metadata Extraction

Extract identity metadata from the directory path:

```text
run_id
backend
case
dataset
concurrency
rep
```

Load the nearest parent metadata file:

```text
<run_root>/<backend>/<case>/<dataset>/run_meta.json
```

Merge these fields when present:

```text
model
case_name
image
nodes
prefill_tp
decode_tp
prefill_dp
decode_dp
prefill_base_gpu_id
decode_base_gpu_id
prefill_resolved_gpu_count
decode_resolved_gpu_count
prefill_gpu_range
decode_gpu_range
total_gpus_per_node
concurrency_values
repetitions
min_prompts
prompts_per_concurrency
cache_mode
output_details
timestamp_utc
backend_order
server_extra_args
```

Also attach paths to sidecar files:

```text
run_meta_path
fingerprint_path
env_path
commands_path
health_before_path
health_after_path
bench_log_path
failed_server_logs_path
```

These paths should be ordinary CSV string columns so a report can link back to
raw evidence.

## Fingerprint and Environment Enrichment

The parser should not merely check that `fingerprint.txt` exists. It should
extract important comparison fields when possible:

```text
git_sha
python_version
torch_version
sglang_kernel_version
sglang_router_version
nixl_version
nixl_cu13_version
mooncake_cuda13_version
cuda_driver_version
gpu_model_summary
ib_device_summary
```

Exact fingerprint format may drift, so this extraction should be best-effort.
Missing extracted fields should become warnings, not hard failures, unless the
strict mode requests them.

The analysis must verify that comparable rows in the same group do not mix:

```text
model
image
git_sha
server_extra_args
case
dataset
cache_mode
nodes
```

If any of those differ inside a backend comparison group, the pairwise row
should be marked invalid and excluded from winner/SLO decisions unless the user
passes an explicit override.

## Canonical Schema

The JSONL keys from `bench_serving` should be normalized immediately. Downstream
CSVs, plots, and reports should use only canonical names.

Required raw-to-canonical mapping:

```text
duration                         -> benchmark_duration_s
completed                        -> successful_requests
total_input_tokens               -> input_tokens
total_input_text_tokens          -> input_text_tokens
total_input_vision_tokens        -> input_vision_tokens
total_output_tokens              -> output_tokens
total_output_tokens_retokenized  -> output_tokens_retokenized
request_throughput               -> request_throughput_req_s
input_throughput                 -> input_token_throughput_tok_s
output_throughput                -> output_token_throughput_tok_s
output_throughput_retokenized    -> output_token_throughput_retokenized_tok_s
total_throughput                 -> total_token_throughput_tok_s
total_throughput_retokenized     -> total_token_throughput_retokenized_tok_s
mean_e2e_latency_ms              -> mean_e2e_latency_ms
median_e2e_latency_ms            -> median_e2e_latency_ms
std_e2e_latency_ms               -> std_e2e_latency_ms
p90_e2e_latency_ms               -> p90_e2e_latency_ms
p99_e2e_latency_ms               -> p99_e2e_latency_ms
mean_ttft_ms                     -> mean_ttft_ms
median_ttft_ms                   -> median_ttft_ms
std_ttft_ms                      -> std_ttft_ms
p99_ttft_ms                      -> p99_ttft_ms
mean_tpot_ms                     -> mean_tpot_ms
median_tpot_ms                   -> median_tpot_ms
std_tpot_ms                      -> std_tpot_ms
p99_tpot_ms                      -> p99_tpot_ms
mean_itl_ms                      -> mean_itl_ms
median_itl_ms                    -> median_itl_ms
std_itl_ms                       -> std_itl_ms
p95_itl_ms                       -> p95_itl_ms
p99_itl_ms                       -> p99_itl_ms
max_itl_ms                       -> max_itl_ms
max_output_tokens_per_s          -> max_output_tokens_per_s
max_concurrent_requests          -> max_concurrent_requests
concurrency                      -> observed_concurrency
errors                           -> errors
```

Keep the requested concurrency from the filename as:

```text
concurrency
```

Keep the JSONL-reported concurrency as:

```text
observed_concurrency
```

If they differ, emit a data-quality warning. This catches accidental file
renames or benchmark argument mismatches.

## Required Derived Fields

Add these fields to `benchmark_results_jsonl.csv`:

```text
expected_requests
success_rate
failed_requests
has_errors
error_count
input_tokens_per_request
output_tokens_per_request
total_tokens
total_tokens_per_request
tokens_per_successful_request
backend_dataset_case
comparison_group
```

Definitions:

- `expected_requests`: prefer JSONL `num_prompts`; otherwise use
  `run_meta` prompt scaling fields or `successful_requests` as fallback.
- `success_rate`: `successful_requests / expected_requests`.
- `failed_requests`: `expected_requests - successful_requests`.
- `has_errors`: true when JSONL `errors` contains any non-empty value.
- `error_count`: count of non-empty error entries.
- `comparison_group`: stable key using model, case, dataset, concurrency,
  cache mode, nodes, image, git SHA, and server args.

For `radixcache`, add:

```text
gsp_num_groups
gsp_prompts_per_group
gsp_system_prompt_len
gsp_question_len
gsp_output_len
```

These can be read from JSONL fields if present, command text if available, or
derived from `num_prompts` using the runner policy.

## Handling `output_details`

When `OUTPUT_DETAILS=1`, JSONL records may include large arrays:

```text
input_lens
output_lens
ttfts
itls
generated_texts
errors
```

The analysis should exploit these arrays instead of ignoring them:

- Recompute request-level p50, p90, p95, p99, min, max for TTFT.
- Recompute ITL distribution metrics from the flattened `itls`.
- Compute request output length distribution.
- Count and summarize non-empty errors.
- Optionally write compact request-level sidecar CSVs.

Do not copy `generated_texts` into CSV. They are large and usually irrelevant
for performance analysis. If a debug export needs them, put them in a separate
artifact and gate it behind an explicit flag.

When `OUTPUT_DETAILS=0`, use aggregate metrics from the JSONL record only.
The report should state whether each row came from detailed or aggregate-only
records.

## Data Quality Checks

Create `csv/data_quality.csv` with one row per issue. Suggested columns:

```text
severity
run_id
backend
case
dataset
concurrency
rep
path
issue_code
message
```

Minimum issue checks:

- Missing `run_meta.json`.
- Missing `fingerprint.txt`.
- Missing JSONL file for an expected rep.
- Empty JSONL file.
- Malformed JSON line.
- More than one JSONL record in a file.
- Missing required metric key.
- Missing required metadata key.
- `successful_requests < expected_requests`.
- `success_rate < 1.0`.
- Non-empty `errors`.
- Filename concurrency differs from JSONL `max_concurrency` or
  `observed_concurrency`.
- Fewer than expected reps for a completed main run.
- Missing backend pair for a comparison group.
- Mixed model/image/git SHA/server args inside a comparison group.
- Mixed cache mode inside a comparison group.
- Mixed `output_details` inside a comparison group.
- Failed server logs exist for a batch that also has JSONL output.

Severity levels:

```text
fatal: invalidates this row or comparison
warn: row usable but should be inspected
info: useful annotation
```

The script should support:

```text
--strict
--allow-partial
--expected-reps 5
```

In strict mode, fatal issues should exit non-zero.

## Per-Run CSV

Produce:

```text
csv/benchmark_results_jsonl.csv
```

This should contain one row per selected JSONL record.

Recommended column order:

```text
run_id
backend
case
dataset
concurrency
rep
model
image
git_sha
nodes
cache_mode
output_details
server_extra_args
prefill_tp
decode_tp
prefill_dp
decode_dp
prefill_gpu_range
decode_gpu_range
expected_requests
successful_requests
success_rate
failed_requests
has_errors
benchmark_duration_s
request_throughput_req_s
input_token_throughput_tok_s
output_token_throughput_tok_s
total_token_throughput_tok_s
mean_ttft_ms
median_ttft_ms
std_ttft_ms
p99_ttft_ms
mean_tpot_ms
median_tpot_ms
std_tpot_ms
p99_tpot_ms
mean_itl_ms
median_itl_ms
std_itl_ms
p95_itl_ms
p99_itl_ms
max_itl_ms
max_output_tokens_per_s
jsonl_path
bench_log_path
fingerprint_path
run_meta_path
```

Append additional known JSONL fields after this stable prefix. Do not discard
unknown JSONL keys; preserve them with a `raw_` prefix if they are scalar.

## Summary CSV

Produce:

```text
csv/benchmark_summary.csv
```

Group by:

```text
model
image
git_sha
case
dataset
backend
concurrency
cache_mode
nodes
server_extra_args
```

For each selected metric, compute:

```text
mean
median
std
min
max
count
coeff_var
```

Primary metrics:

```text
request_throughput_req_s
output_token_throughput_tok_s
total_token_throughput_tok_s
mean_ttft_ms
median_ttft_ms
p99_ttft_ms
mean_tpot_ms
median_tpot_ms
p99_tpot_ms
mean_itl_ms
median_itl_ms
p95_itl_ms
p99_itl_ms
max_itl_ms
success_rate
```

Use medians as the primary comparison numbers in reports and plots. Means and
standard deviations should support stability interpretation, not replace
median-based comparison.

Coefficient of variation should be computed as:

```text
std / abs(mean)
```

If the mean is zero or null, leave coefficient of variation null.

## Pairwise Backend Comparison

Produce:

```text
csv/backend_pairwise.csv
```

One row per comparable pair:

```text
model
image
git_sha
case
dataset
concurrency
cache_mode
nodes
server_extra_args
```

Required columns:

```text
nixl_output_token_throughput_tok_s_median
mooncake_output_token_throughput_tok_s_median
output_throughput_ratio_nixl_over_mooncake
nixl_total_token_throughput_tok_s_median
mooncake_total_token_throughput_tok_s_median
total_throughput_ratio_nixl_over_mooncake
nixl_request_throughput_req_s_median
mooncake_request_throughput_req_s_median
request_throughput_ratio_nixl_over_mooncake
nixl_p99_ttft_ms_median
mooncake_p99_ttft_ms_median
p99_ttft_ratio_nixl_over_mooncake
nixl_p99_tpot_ms_median
mooncake_p99_tpot_ms_median
p99_tpot_ratio_nixl_over_mooncake
nixl_p99_itl_ms_median
mooncake_p99_itl_ms_median
p99_itl_ratio_nixl_over_mooncake
nixl_success_rate_median
mooncake_success_rate_median
winner_throughput
winner_latency
winner_slo
comparison_valid
invalid_reason
```

Ratio conventions:

- Throughput ratio greater than 1 means NIXL is faster.
- Latency ratio less than 1 means NIXL is lower latency.
- Do not declare a winner when either backend has incomplete reps, failed
  requests, missing metrics, or incompatible metadata.

## SLO Summary

Produce:

```text
csv/slo_summary.csv
```

Inputs should be configurable:

```text
--slo-p99-ttft-ms 5000
--slo-p99-tpot-ms 200
--slo-p99-itl-ms 200
--slo-success-rate 1.0
```

Group by:

```text
model
image
git_sha
case
dataset
backend
cache_mode
nodes
server_extra_args
```

For each group, find the highest concurrency satisfying all SLOs. Output:

```text
max_slo_concurrency
output_token_throughput_tok_s_at_slo
total_token_throughput_tok_s_at_slo
request_throughput_req_s_at_slo
p99_ttft_ms_at_slo
p99_tpot_ms_at_slo
p99_itl_ms_at_slo
ttft_slo_headroom_ms
tpot_slo_headroom_ms
itl_slo_headroom_ms
success_rate_at_slo
first_failed_concurrency
first_failed_reason
```

SLO pass/fail should use the median across repetitions by default. Add a
stricter optional mode that requires every repetition to pass:

```text
--slo-mode median|all-reps
```

This table is the most important decision artifact because it answers:

```text
Which backend serves more traffic before violating latency?
```

## Plot Corrections

The existing plots should be expanded beyond two panels.

Generate per `model x case x dataset x cache_mode`:

```text
output_token_throughput_tok_s vs concurrency
total_token_throughput_tok_s vs concurrency
request_throughput_req_s vs concurrency
p99_ttft_ms vs concurrency
p99_tpot_ms vs concurrency
p99_itl_ms vs concurrency
success_rate vs concurrency
SLO pass/fail vs concurrency
```

Plot rules:

- Use log2 x-axis for concurrency.
- Draw one line per backend.
- Plot median as the line.
- Show min/max or standard deviation as error bands when count is at least 3.
- Mark missing/incomplete points visibly, or omit them and report omissions in
  `data_quality.csv`.
- Draw SLO thresholds as horizontal lines on latency plots.
- Include cache mode, server args, model, git SHA, and case in plot title or
  subtitle.

Useful combined figures:

- One summary page per case/dataset with throughput and p99 latency panels.
- One SLO comparison plot per case/dataset showing pass/fail by backend.
- One heatmap of winner by case/dataset/concurrency.

## Markdown Report

Generate:

```text
report.md
```

Minimum sections:

1. Run metadata:
   - analyzed run roots
   - model
   - git SHA
   - image
   - package versions
   - node count
   - server extra args
   - cache mode
2. Data completeness:
   - expected rows
   - observed rows
   - missing reps
   - malformed JSONL files
   - failed requests
   - missing backend pairs
3. SLO summary:
   - best backend by case/dataset
   - max SLO concurrency
   - throughput at max SLO concurrency
4. Pairwise backend comparison:
   - median throughput ratios
   - p99 latency ratios
   - winner fields
5. Dataset-specific findings:
   - rand
   - sharegpt
   - radixcache warm
   - radixcache cold, when available
6. Case-specific findings:
   - symmetric TP
   - asymmetric prefill/decode TP
7. Outliers and instability:
   - high coefficient of variation
   - failed requests
   - rows with multi-record JSONL files
8. Links:
   - CSV outputs
   - plots
   - JSONL roots
   - failed server logs

The report should avoid claiming a backend is better when data quality checks
invalidate the comparison.

## Backward Compatibility

Keep the current text-log pipeline as legacy:

```text
input/*.log
  -> cleaned_input/*_cleaned.log
  -> csv/benchmark_results.csv
  -> result_png/*_comparison.png
```

Rename its role in docs to:

```text
Legacy text-log parser for historical runs.
```

Do not remove it unless all historical input data has been migrated. New runs
should use:

```text
--output-file "$jsonl"
```

and should be analyzed through the JSONL pipeline.

## Implementation Phases

### Phase A: JSONL Flattening

Implement recursive discovery, JSONL parsing, metadata merge, canonical schema,
and `benchmark_results_jsonl.csv`.

Done when:

- Phase 0 smoke JSONL files produce a per-run CSV.
- Appended multi-record files are detected.
- Missing metadata and fingerprints are reported.

### Phase B: Quality and Summary

Implement `data_quality.csv` and `benchmark_summary.csv`.

Done when:

- Missing reps and failed requests are visible.
- Summary rows include median, mean, std, min, max, count, and coeff var.
- Strict mode exits non-zero for fatal issues.

### Phase C: Pairwise and SLO

Implement `backend_pairwise.csv` and `slo_summary.csv`.

Done when:

- NIXL/Mooncake comparisons are made only within compatible groups.
- Throughput and latency ratios follow the documented convention.
- Highest SLO-passing concurrency is computed for each backend.

### Phase D: Plots and Report

Implement PNG plots and `report.md`.

Done when:

- Every complete case/dataset/cache-mode group has throughput and latency plots.
- The report includes data completeness, pairwise winners, SLO winners, and
  links to raw evidence.

## Acceptance Criteria

The corrected analysis flow is complete when:

- JSONL is the primary input for new runs.
- Text logs are used only for debug links and legacy fallback.
- Every row preserves run, backend, case, dataset, concurrency, rep, model,
  image, git SHA, cache mode, server args, and artifact paths.
- Current JSONL keys are normalized into stable canonical column names.
- Missing reps, failed requests, errors, mixed metadata, and missing backend
  pairs are visible before plots are trusted.
- Repetition summaries report median, mean, std, min, max, count, and
  coefficient of variation.
- Pairwise backend comparisons include throughput ratios and p99 latency ratios.
- SLO summary identifies the highest concurrency each backend can serve while
  satisfying latency and success-rate requirements.
- Plots show throughput, p99 TTFT, p99 TPOT, p99 ITL, success rate, and SLO
  pass/fail, not only mean TTFT and mean ITL.
- A Markdown report ties the CSVs, plots, fingerprints, JSONL files, and failed
  server logs into one reviewable artifact.
