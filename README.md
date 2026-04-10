# Spark LLM Tuning Loop

This repo now treats every external dependency as a replaceable backend:
- `llm`: `router` or `local`
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
  --iterations 2 \
  --use-base-for-first
```

Artifacts are written to `output/`.

### Spark standalone mode (Docker Compose)

```bash
docker compose run --rm lens-agent-standalone
```

This uses `spark_submit` against a local Spark standalone cluster defined in `docker-compose.yml` and `examples/docker/config.standalone.yaml`.

## Config Shape

```yaml
llm:
  backend: "router"
  router:
    base_url: "https://llm-router.internal"
    api_key_env: "LLM_ROUTER_API_KEY"
    model: "gpt-4o-mini"
    timeout_seconds: 30
    allow_models: ["gpt-4o-mini"]

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
