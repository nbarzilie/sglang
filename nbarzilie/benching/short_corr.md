# Short Analysis Corrections

The analysis workflow should change.

Required modifications:

1. Merge across many `RUN_ID`s. Phase 1 now uses 16 separate sbatch jobs shaped as:

```text
case x dataset
```

Each shard runs both transfer backends in the same Slurm allocation. Shard
submission alternates backend order:

```text
nixl -> mooncake -> mooncake -> nixl -> nixl -> mooncake ...
```

This is a run-order control only. Backend comparison should still pair rows by:

```text
model, backend, case, dataset, concurrency
```

and compare backends by:

```text
model, case, dataset, concurrency
```

2. Treat `radixcache_cold` as a first-class Phase 1 dataset. It is the same
workload family as `radixcache`, but with a different cache policy. Add a
normalized field:

```text
dataset_family = radixcache
cache_policy = cold_each_rep
```

For normal `radixcache`:

```text
dataset_family = radixcache
cache_policy = warm_across_reps
```

3. Read `cache_policy` from `run_meta.json`. The old `CACHE_MODE` field should no longer be required.

4. Expect dataset-derived flushing:

```text
rand            -> cold_each_rep
sharegpt        -> cold_each_rep
radixcache      -> warm_across_reps
radixcache_cold -> cold_each_rep
```

5. Completeness checks should be per shard:

```text
case, dataset -> expected 2 backends x 4 concurrencies x 5 reps
backend, case, dataset, concurrency -> expected 5 reps
```

For Phase 1, expected full coverage is:

```text
2 backends x 4 cases x 4 datasets x 4 concurrencies x 5 reps = 640 rows
```

There is no separate optional cold-radix point in the main design; `radixcache_cold`
is included in the 16 shards.

6. Reports should show cache policy explicitly. Do not mix warm radixcache and cold radixcache rows in the same aggregate unless grouped by `cache_policy`.

7. Request counts are concurrency-dependent and must be treated as workload
metadata, not inferred from concurrency alone:

```text
c2  -> 128 requests
c8  -> 512 requests
c32 -> 1024 requests
c64 -> 1024 requests
```

`c1` is intentionally excluded from the main Phase 1 analysis because it is too
slow and mostly measures serial latency. If old `c1` smoke rows exist, treat them
as a separate smoke/latency sensitivity set, not as missing Phase 1 coverage.

Read `prompt_count_by_concurrency` from `run_meta.json` when available. Always
parse and use actual `successful_requests`, `input_tokens`, `output_tokens`, and
`duration` from JSONL for aggregation.
