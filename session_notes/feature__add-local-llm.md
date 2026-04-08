# Session Notes for `feature/add-local-llm`

## Goal

Add support for running the tuning loop with a local LLM model in Docker, while keeping the architecture compatible with future non-dev usage such as Kubernetes runs against a reachable local-model endpoint.

## What Was Implemented

### 1. New LLM backend: `ollama`

Added a dedicated `ollama` backend instead of overloading the existing `router` backend.

Updated files:
- `src/agent_shell/config.py`
- `src/agent_shell/factory.py`
- `src/agent_shell/ollama.py`

Implemented config fields:
- `llm.ollama.base_url`
- `llm.ollama.model`
- `llm.ollama.timeout_seconds`
- `llm.ollama.keep_alive`
- `llm.ollama.options`

The client talks to Ollama over HTTP via `/api/chat`.

### 2. Docker dev path for local LLM execution

Added:
- `examples/docker/config.ollama.yaml`
- `ollama` service in `docker-compose.yml`
- `lens-agent-ollama` runner in `docker-compose.yml`
- persistent volume `ollama-data`

The Ollama runner now submits Spark jobs to the local Spark standalone cluster:
- `spark://spark-master:7077`

The `lens-agent-ollama` runner depends on:
- `ollama`
- `spark-history`
- `spark-master`
- `spark-worker`

### 3. GPU support for Ollama in Docker

Configured the `ollama` Docker service to request NVIDIA GPU access through Compose using:
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `deploy.resources.reservations.devices`

This relies on host-side NVIDIA driver/toolkit availability.

### 4. Better Docker build hygiene

Added `.dockerignore` so local virtualenv directories and transient files are not sent into Docker build context.

This fixed Docker build failures caused by the broken `.venv/lib64` symlink/path in the repo checkout.

### 5. Reproducible dev environment

Earlier in this branch/session:
- aligned project Python requirement with actual tooling
- updated CI to use `uv sync --extra dev --frozen`
- added test execution to CI
- documented `uv`-based dev setup and fallback instructions

Relevant files:
- `pyproject.toml`
- `.github/workflows/ci.yml`
- `.python-version`
- `README.md`

## Tuning Loop Improvements

### 6. `iterations` moved into YAML config

The number of tuning runs is now driven by:
- `tuning.iterations`

CLI flag:
- `--iterations`

now acts only as an override.

Updated files:
- `src/agent_shell/config.py`
- `main.py`
- all example configs
- `docker-compose.yml`

### 7. Prompt moved into YAML config

The tuning system prompt is now configurable through:
- `tuning.prompt`

Code now uses prompt text from YAML rather than a hardcoded prompt in `main.py`.

Updated files:
- `src/agent_shell/config.py`
- `main.py`
- all example configs
- `README.md`

### 8. Prompt made generic over YAML-defined tuning params

The prompt no longer assumes fixed parameter names like:
- `executor.instances`
- `executor.cores`
- `executor.memory_gb`
- `spark.sql.shuffle.partitions`

Instead, the user prompt now includes:
- `tunable_params`

with each parameter's:
- type
- min
- max
- default
- manifest path

This makes the tuning flow generic for arbitrary parameters defined in `tuning.params`.

### 9. Driver logs removed from LLM prompt

Previously the entire `run_meta`, including `driver_logs`, was appended into `history` and sent to the model.

That was changed so the LLM only receives a compact history entry containing:
- params
- rationale
- application_state
- runtime_seconds
- requested_gb
- requested_gb_seconds
- spill_gb
- output_files
- small_files
- spark_ui
- history_api

Artifacts still retain the full driver logs on disk in `output/`.

### 10. Duplicate configuration guardrail

Added code to prevent the same parameter configuration from being run twice.

If the model proposes a configuration that has already been seen:
- the code detects the duplicate using only tunable parameters
- attempts to move to the nearest valid unique neighbor
- raises an error only if no unique alternative can be found

This logic lives in `main.py`.

### 11. LLM JSON retry logic with YAML-configurable retry count

Added:
- `tuning.llm_json_retries`

If the model returns invalid JSON:
- a retry is triggered
- the next request includes `retry_feedback`
- the feedback contains:
  - attempt number
  - JSON parse error
  - excerpt from the invalid response
  - instruction to return strict valid JSON

This is now configurable in YAML and defaults to `2`.

## Logging Improvements

### 12. Ollama request/response logging

Added logging in `src/agent_shell/ollama.py` for:
- request target URL
- request payload
- waiting for response
- response arrival

The payload logging is currently very verbose because it includes the full prompt JSON. This may still be worth reducing later.

### 13. Better timeout handling for local models

Added explicit timeout error handling for Ollama requests so failures now produce a more actionable message:
- increase `llm.ollama.timeout_seconds`
- or use a smaller/faster model

Also increased the example Ollama timeout in dev config and reduced `num_predict`.

## Config Changes

### `examples/docker/config.ollama.yaml`

Current important behavior:
- uses `llm.backend=ollama`
- uses Spark standalone cluster, not `local[*]`
- has narrowed tuning bounds for dev experimentation
- currently sets:
  - `tuning.iterations: 5`
  - `tuning.llm_json_retries: 2`

### Other configs updated

Updated to include new tuning keys:
- `config.yaml`
- `examples/local/config.local.yaml`
- `examples/docker/config.docker.yaml`
- `examples/docker/config.standalone.yaml`

## Tests Added or Updated

Updated test coverage in:
- `tests/test_config.py`
- `tests/test_local_run.py`

Covered areas:
- loading `ollama` config
- building `OllamaLlmClient`
- parsing Ollama chat response
- loading custom prompt from YAML
- loading `tuning.iterations`
- loading `tuning.llm_json_retries`
- compact LLM history generation
- duplicate-config resolution
- retry behavior after invalid JSON from model

Latest local verification completed:
- `ruff check`
- `pytest`
- `compileall`

At the end of the work, tests were passing with:
- `9 passed`

## Runtime Observations

### Observed behavior of local model

During real Docker runs with `qwen2.5:3b`:
- the model sometimes returned invalid JSON
- this is why retry logic was added
- the model also tended to overfit to or ignore certain parameters depending on prompt wording

### Spark History behavior

When the runner was using `local[2]`, Spark History showed local runs from inside the container.

This was changed so `lens-agent-ollama` now runs through Spark standalone instead.

## Remaining Notes / Risks

1. The Ollama payload log is still too large in practice because it prints the full prompt JSON.
2. The prompt policy is now editable through YAML, so behavior can drift if the prompt is heavily changed.
3. There is still one hardcoded parameter-specific behavior in `main.py`:
   - setting `spark.kubernetes.executor.request.cores` when `executor.cores` exists
   - this is still not fully generic over arbitrary tuning params
4. Full end-to-end validation for every config variant has not been exhaustively rerun after every prompt tweak, though unit tests and compose validation are passing.

## Recommended Next Steps

1. Reduce Ollama payload logging to a concise summary instead of dumping the full JSON request.
2. Consider making the duplicate-resolution strategy more exploration-aware, so it prefers parameters that have not changed yet across runs.
3. Consider removing the remaining parameter-specific `executor.cores` special case from `main.py` and pushing that behavior into config.
