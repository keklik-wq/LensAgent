# Plan for `feature/add-local-llm`

1. Add a new `llm.backend`, preferably a dedicated `ollama` backend rather than overloading `router`. This keeps the local-model path usable for Docker dev now and reusable later for `kubectl` runs through a normal network endpoint.

2. Implement `OllamaLlmClient` and config support in:
   - `src/agent_shell/config.py`
   - `src/agent_shell/factory.py`

   The config should include fields such as `base_url`, `model`, `timeout_seconds`, and optionally `keep_alive` or model `options`.

3. Add a Docker dev path for local LLM execution in:
   - `docker-compose.yml`
   - `examples/docker/config.ollama.yaml`

   This should include a dedicated `ollama` service, a separate runner such as `lens-agent-local-llm`, and a persistent volume for models. Prefer using a profile so the heavier stack does not start by default.

4. Update the runbook and usage docs in `README.md`.
   Document the commands to pull a model and run the agent with a local LLM in Docker. Also make it explicit that this backend is not dev-only and can later be reused from Kubernetes if an endpoint is reachable.

5. Add minimum test coverage in `tests/test_config.py`.
   At minimum, cover loading the new backend config and client factory wiring. If time allows, add a focused unit test for response parsing in the new client.
