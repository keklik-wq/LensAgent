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

## Local Docker Compose

```bash
docker compose run --rm lens-agent
```

That command runs:
```bash
python main.py \
  --manifest examples/docker/sparkapp.yaml \
  --transform examples/docker/job.py \
  --config examples/docker/config.docker.yaml \
  --use-base-for-first
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
uv sync --extra dev
```

That creates a reproducible `.venv` from [`uv.lock`](./uv.lock) and installs lint/test tooling.

If `.venv` already exists from another OS or an older toolchain and `uv sync` cannot reuse it, create a clean environment in a separate directory:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev --frozen
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

### Spark standalone mode (Docker Compose)

```bash
docker compose run --rm lens-agent-standalone
```

This uses `spark_submit` against a local Spark standalone cluster defined in `docker-compose.yml` and `examples/docker/config.standalone.yaml`.

## Config Shape

```yaml
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

Legacy `llm_router:` config is still accepted and normalized to the new structure.
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
