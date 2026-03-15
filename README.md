# Spark LLM Tuning Loop

This repo runs an LLM‑driven tuning loop for Spark jobs on Kubernetes using a SparkApplication manifest.

**Flow**
1. Apply a SparkApplication manifest via `kubectl`.
2. Wait for completion.
3. Pull metrics from Spark History Server.
4. Send run history to an LLM and get the next parameter proposal.
5. Repeat for N iterations.

All outputs (manifests, masked manifests, run metadata, summary) are stored under `output/`.

## Quick Start
Requires Python 3.14.2 (venv already initialized by you).
```bash
uv venv .venv
uv pip install -e .
export LLM_ROUTER_API_KEY=your-token
python3 main.py \
  --manifest input/UVIM_SignalsLinking_partners_research.yaml \
  --transform input/UVIM_SignalsLinking_partners_research.py \
  --config config.yaml \
  --history-url https://your-history-server \
  --iterations 5 \
  --use-base-for-first
```

## Config (`config.yaml`)
Only the LLM router settings are required:
```yaml
llm_router:
  base_url: "https://llm-router.internal"
  api_key_env: "LLM_ROUTER_API_KEY"
  model: "gpt-4o-mini"
  timeout_seconds: 30
  allow_models:
    - "gpt-4o-mini"
```

## Outputs
- `output/runs/run_XXX/manifest_XXX.yaml`
- `output/runs/run_XXX/manifest_XXX.masked.yaml`
- `output/runs/run_XXX/run_XXX.json`
- `output/summary.json`

## Notes
- The loop uses the Spark History Server stages API: `/api/v1/applications/{appId}/stages`.
- The app ID is taken from SparkApplication status or parsed from driver logs.
- Parameter tuning targets:
  - `spark.sql.shuffle.partitions`
  - `executor.cores`
  - `executor.instances`
  - `executor.memory_gb`

Example History API payload is in `example_parts/stages.txt`.
