# Spark LLM Tuning Loop

This repo now treats every external dependency as a replaceable backend:
- `llm`: `router`, `local`, or `ollama`
- `spark_runtime`: `kubernetes`, `spark_submit`, or `local`
- `spark_history`: `http` or `local`
- `tuning`: configurable parameter map for LLM-tuned Spark settings

The tuning loop no longer depends directly on `kubectl` or on the Kubernetes Python client at import time. Kubernetes support still exists, but it is just one runtime adapter.

## Backends

### Production-style run
Use:
- `llm.backend=router`
- `spark_runtime.backend=kubernetes`
- `spark_history.backend=http`

Install Kubernetes support only when you need it:
```bash
uv pip install -e ".[kubernetes]"
```

### Local run
Use:
- `llm.backend=local`
- `spark_runtime.backend=local`
- `spark_history.backend=local`

This mode runs end-to-end without Kubernetes and is intended for local development, CI smoke checks, and parallel agent work.

### Local model run
Use:
- `llm.backend=ollama`
- `spark_runtime.backend=spark_submit` or `kubernetes`
- `spark_history.backend=http` or `local`

This backend targets a reachable Ollama HTTP endpoint, so it works for Docker dev now and remains usable later from Kubernetes.

## Kubernetes Run

Use this mode when you want the tuning loop to submit `SparkApplication` resources to a Kubernetes cluster through the Spark Operator.

Prerequisites:
- a working kubeconfig or in-cluster Kubernetes credentials
- Spark Operator installed in the target cluster
- a reachable Spark History Server URL for `spark_history.http.base_url`
- an LLM endpoint reachable from the tuning process

Install the Kubernetes runtime dependency:
```bash
uv pip install -e ".[kubernetes]"
```

Choose one of these LLM backends:
- `llm.backend=router` for a remote router/OpenAI-compatible service
- `llm.backend=ollama` for an Ollama endpoint reachable from where `main.py` runs

Minimal config shape:
```yaml
run:
  manifest: "path/to/sparkapp.yaml"
  transform: "path/to/job.py"
  first_run_mode: "base"

llm:
  backend: "router"
  router:
    base_url: "https://llm-router.internal"
    chat_path: "/v1/chat/completions"
    api_key_env: "LLM_ROUTER_API_KEY"
    model: "gpt-4o-mini"
    timeout_seconds: 30
    allow_models: ["gpt-4o-mini"]

spark_runtime:
  backend: "kubernetes"
  kubernetes:
    kube_context: "my-context"
    kubeconfig_path: "/path/to/kubeconfig"

spark_history:
  backend: "http"
  http:
    base_url: "https://spark-history.internal"
    timeout_seconds: 30

tuning:
  iterations: 5
  llm_json_retries: 2
  prompt: |
    You are a Spark tuning assistant.
    Return ONLY valid JSON that matches the schema exactly.
  params:
    spark.sql.shuffle.partitions:
      path: "spec.sparkConf.spark.sql.shuffle.partitions"
      type: "int"
      min: 100
      max: 2000
```

Run from your workstation or from a pod with cluster access:
```bash
python main.py \
  --config path/to/config.yaml \
  --namespace my-namespace
```

Useful optional flags:
- `--kube-context` overrides `spark_runtime.kubernetes.kube_context`
- `--history-url` overrides `spark_history.http.base_url`
- `--iterations` overrides `tuning.iterations`
- `--max-total-memory-gb` overrides `tuning.constraints.total_memory_gb.max`
- `--driver-container` selects the driver container when driver logs are needed

How it works in Kubernetes mode:
- `main.py` loads the base SparkApplication manifest
- tuned parameters are applied to a per-run manifest
- the Kubernetes runtime creates or replaces the `SparkApplication` resource
- the loop waits for completion, fetches stage data from Spark History Server, and feeds the summarized history back into the LLM for the next iteration

Notes:
- `llm.router.base_url` is the router host/root URL
- `llm.router.chat_path` is the relative API path used for chat requests
- `kube_context` is the name of a context inside the kubeconfig, not a file path
- `kubeconfig_path` is an optional explicit path to the kubeconfig file
- if `kubeconfig_path` is omitted, the Kubernetes Python client uses the default kubeconfig location
- if loading local kubeconfig fails, the runtime falls back to in-cluster authentication

## Local Docker Compose

```bash
docker compose run --rm lens-agent
```

That command runs:
```bash
python main.py \
  --config examples/docker/config.docker.yaml
```

Artifacts are written to `output/`.

### Local Docker Compose with Ollama

Start the local model service:
```bash
docker compose --profile ollama up -d ollama spark-history
```

The `ollama` service is configured to request all NVIDIA GPUs through Docker Compose. Per Docker's GPU reservation rules, this requires a working NVIDIA driver and container toolkit on the host.

Pull a model into the persistent Ollama volume:
```bash
docker compose --profile ollama exec ollama ollama pull qwen2.5:3b
```

Run the tuning loop against the local model:
```bash
docker compose --profile ollama run --rm lens-agent-ollama
```

This path uses [`examples/docker/config.ollama.yaml`](./examples/docker/config.ollama.yaml).
The default Ollama dev config uses a longer timeout because first-token latency on CPU can easily exceed a minute.
The Ollama dev runner submits Spark jobs to the local Spark standalone cluster at `spark://spark-master:7077`, not to `local[*]`.

## Dev Environment

Primary path: `uv` with the checked-in lockfile.

```bash
uv python install 3.10
uv sync --extra dev --frozen
```

That creates a reproducible `.venv` from [`uv.lock`](./uv.lock) and installs lint/test tooling.

If `.venv` already exists from another OS or an older toolchain and `uv sync` cannot reuse it, remove it and recreate the same `.venv`:

```bash
rm -rf .venv
uv sync --extra dev --frozen
```

Run checks with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Fallback when `uv` is unavailable:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

If you need to recreate the environment in PowerShell:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --extra dev --frozen
```

### Spark standalone mode (Docker Compose)

```bash
docker compose run --rm lens-agent-standalone
```

This uses `spark_submit` against a local Spark standalone cluster defined in `docker-compose.yml` and `examples/docker/config.standalone.yaml`.

## Config Shape

```yaml
run:
  manifest: "examples/docker/sparkapp.yaml"
  transform: "examples/docker/job.py"
  first_run_mode: "base"

llm:
  backend: "ollama"
  ollama:
    base_url: "http://ollama:11434"
    model: "qwen2.5:3b"
    timeout_seconds: 90
    keep_alive: "30m"
    options:
      num_predict: 256

spark_runtime:
  backend: "kubernetes"
  kubernetes:
    kube_context: null
    kubeconfig_path: null

spark_history:
  backend: "http"
  http:
    base_url: "https://spark-history.internal"
    timeout_seconds: 30

tuning:
  iterations: 2
  prompt: |
    You are a Spark tuning assistant.
    ...
  params:
    spark.sql.shuffle.partitions:
      path:
        - "spec"
        - "sparkConf"
        - "spark.sql.shuffle.partitions"
      type: "int"
      min: 200
      max: 10000
    executor.cores:
      path: "spec.executor.cores"
      type: "int"
      min: 1
      max: 16
    executor.instances:
      path: "spec.executor.instances"
      type: "int"
      min: 1
      max: 500
    executor.memory_gb:
      path: "spec.executor.memory"
      type: "memory_gb"
      min: 1
      max: 256
  constraints:
    total_memory_gb:
      max: 500
```

The run inputs now come from `run.manifest`, `run.transform`, and `run.first_run_mode`.
The number of tuning-loop runs now defaults from `tuning.iterations`, and `--iterations` only overrides it.
The tuning system prompt is configurable through `tuning.prompt`.

## Why this refactor

- `main.py` composes adapters instead of instantiating Kubernetes and HTTP dependencies inline.
- Local execution no longer needs the `kubernetes` package.
- Replacing one environment with another is now a config change, not a rewrite.

## Adding a new tuning parameter

Add an entry under `tuning.params`. For example, to tune `spark.memory.fraction`:
```yaml
tuning:
  params:
    spark.memory.fraction:
      path:
        - "spec"
        - "sparkConf"
        - "spark.memory.fraction"
      type: "float"
      min: 0.1
      max: 0.9
```
