# Spark LLM Tuning Loop

This repo now treats every external dependency as a replaceable backend:
- `llm`: `router` or `local`
- `spark_runtime`: `kubernetes` or `local`
- `spark_history`: `http` or `local`

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
docker compose up --build
```

That command runs:
```bash
python main.py \
  --manifest examples/local/sparkapp.yaml \
  --transform examples/local/job.py \
  --config examples/local/config.local.yaml \
  --iterations 2 \
  --use-base-for-first
```

Artifacts are written to `output/`.

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
```

Legacy `llm_router:` config is still accepted and normalized to the new structure.

## Why this refactor

- `main.py` composes adapters instead of instantiating Kubernetes and HTTP dependencies inline.
- Local execution no longer needs the `kubernetes` package.
- Replacing one environment with another is now a config change, not a rewrite.
