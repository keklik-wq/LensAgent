# Spark LLM Agent Shell

Minimal, policy‑driven shell around an LLM that can analyze Spark jobs using Spark History Server, Kubernetes logs, and output data checks. All external interactions are wrapped in code and strictly allow‑listed.

## What this provides
- A safe runtime for the LLM (router client + prompt builder).
- Strict capability wrappers: Spark History Server, K8s logs, output storage inspection.
- Action validation and execution with explicit allowlists.
- A single orchestrator entry point you can extend.

## Quick start
```bash
uv pip install -e ".[dev]"
python3 main.py
```

## .env example
```bash
CONFIG_PATH=config.yaml
APP_ID=app-20240311123456-0001
K8S_NAMESPACE=spark-jobs
OUTPUT_PATH=/data/warehouse/my_table
```

## Directory layout
- `main.py` — CLI entry point.
- `config.yaml` — default config with placeholders.
- `src/agent_shell/` — core implementation.

## Notes
- This is a scaffold. Replace stub adapters with your internal APIs.
- The LLM never accesses the system directly; it only sees structured context.

## Spark Tuning Loop (LLM-driven)
This project includes an automated tuning loop that:
1. Generates a SparkApplication manifest.
2. Applies it with `kubectl`.
3. Waits for completion.
4. Collects runtime + spill metrics from Spark History Server (`/api/v1/applications/{appId}/stages`).
5. Sends the run history to the LLM to propose the next set of parameters.
6. Repeats for N iterations.

### Prerequisites
- `kubectl` configured and pointing at the right cluster/context.
- SparkOperator CRD `sparkapplications.sparkoperator.k8s.io` is installed.
- Spark History Server is reachable (no auth).
- LLM router config is set in `config.yaml` (see `llm_router` block).

### Inputs
- Transformation code file (e.g. `input/UVIM_SignalsLinking_partners_research.py`)
- Base SparkApplication manifest (e.g. `input/UVIM_SignalsLinking_partners_research.yaml`)
- `config.yaml` with LLM router settings
- `.env` with iteration count:
  - `ITERATIONS=5`

### Run The Full Loop
```bash
python experiment.py iterate \
  --manifest input/UVIM_SignalsLinking_partners_research.yaml \
  --transform input/UVIM_SignalsLinking_partners_research.py \
  --config config.yaml \
  --history-url https://hs.dev.bdp-common.k8s.mediascope.net \
  --use-base-for-first
```

If driver logs are in a non-default container:
```bash
python experiment.py iterate ... --driver-container <name>
```

### What Gets Tuned
The LLM proposes the next values for:
- `spark.sql.shuffle.partitions`
- `executor.cores`
- `executor.instances`
- `executor.memory_gb`

Guardrails are enforced in code:
- Total requested memory (driver + executors) must be <= 500 GB.
- Min/Max ranges are clamped to safe defaults.

### How Metrics Are Computed
From the History Server stages API:
- `runtime_seconds`: `max(completionTime) - min(submissionTime)`
- `spill_gb`: sum of `memoryBytesSpilled + diskBytesSpilled` across stages
- `requested_gb_seconds`: `(driver_memory_gb + executor_instances * executor_memory_gb) * runtime_seconds`

Best run is the one with the smallest `requested_gb_seconds`.

### Output
All runs are saved under `output/`:
- `output/runs/run_XXX/manifest_XXX.yaml` (full manifest)
- `output/runs/run_XXX/manifest_XXX.masked.yaml` (secrets masked)
- `output/runs/run_XXX/run_XXX.json` (metrics + parameters)
- `output/summary.json` (best run summary)

Each run JSON includes `is_best: true` for the best run.

### Secret Masking
The masked manifest replaces values for keys containing:
`secret`, `password`, `access.key`, `secret.key`, `api_key`, `token`, `keytab`

### Notes
- The loop derives `appId` from the SparkApplication status, and falls back to driver logs.
- The Spark UI link is saved as:
  `https://<history-host>/history/{appId}/stages/`

### LLM Request Example (Local)
You can see a local request example in `request_llm.py`. This script is not used by the tuning loop,
but it shows how to send prompts to your LLM endpoint directly.
