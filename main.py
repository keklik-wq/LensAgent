from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from src.agent_shell import AgentShell, AppContext, ShellConfig


def main() -> None:
    load_dotenv()
    config_path = os.getenv("CONFIG_PATH")
    app_id = os.getenv("APP_ID")
    if not config_path or not app_id:
        raise SystemExit("CONFIG_PATH and APP_ID must be set in .env")
    namespace = os.getenv("K8S_NAMESPACE")
    output_path = os.getenv("OUTPUT_PATH")

    config = ShellConfig.load(config_path)
    namespace = namespace or config.k8s.namespace
    shell = AgentShell(config)
    result = shell.run(
        AppContext(
            app_id=app_id,
            namespace=namespace,
            output_path=output_path,
        )
    )
    print(
        json.dumps(
            {
                "summary": result.summary,
                "proposals": [p.__dict__ for p in result.proposals],
                "diagnostics": result.diagnostics,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
