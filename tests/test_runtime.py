from src.agent_shell.runtime import KubernetesSparkRuntime


class _FakeApiException(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status={status}")
        self.status = status


class _FakeCustomObjectsApi:
    def __init__(self, existing: dict[str, object] | None = None) -> None:
        self.existing = existing
        self.replace_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []

    def get_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
    ) -> dict[str, object]:
        del group, version, namespace, plural, name
        if self.existing is None:
            raise _FakeApiException(404)
        return self.existing

    def replace_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
        body: dict[str, object],
    ) -> None:
        del group, version, namespace, plural, name
        self.replace_calls.append(body)

    def create_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        body: dict[str, object],
    ) -> None:
        del group, version, namespace, plural
        self.create_calls.append(body)

    def delete_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
    ) -> None:
        self.delete_calls.append(
            {
                "group": group,
                "version": version,
                "namespace": namespace,
                "plural": plural,
                "name": name,
            }
        )


class _FakeCoreApi:
    def __init__(self, pod_name: str = "demo-driver", logs: str = "") -> None:
        self._pod_name = pod_name
        self._logs = logs

    def list_namespaced_pod(self, namespace: str, label_selector: str | None = None):
        del namespace, label_selector
        metadata = type("FakeMetadata", (), {"name": self._pod_name})()
        item = type("FakePod", (), {"metadata": metadata})()
        return type("FakePodList", (), {"items": [item]})()

    def read_namespaced_pod_log(
        self,
        name: str,
        namespace: str,
        container: str | None = None,
        tail_lines: int | None = None,
    ) -> str:
        del name, namespace, container, tail_lines
        return self._logs


def test_apply_replace_uses_existing_resource_version() -> None:
    runtime = object.__new__(KubernetesSparkRuntime)
    runtime._custom_api = _FakeCustomObjectsApi(existing={"metadata": {"resourceVersion": "12345"}})
    runtime._k8s_client = type(
        "FakeK8sClient",
        (),
        {"exceptions": type("FakeExceptions", (), {"ApiException": _FakeApiException})},
    )()

    manifest = {"metadata": {"name": "demo"}}

    runtime._apply("sparkoperator.k8s.io", "v1beta2", "spark", "demo", manifest)

    assert manifest["metadata"]["resourceVersion"] == "12345"
    assert runtime._custom_api.replace_calls == [manifest]


def test_apply_creates_when_object_is_missing() -> None:
    runtime = object.__new__(KubernetesSparkRuntime)
    runtime._custom_api = _FakeCustomObjectsApi(existing=None)
    runtime._k8s_client = type(
        "FakeK8sClient",
        (),
        {"exceptions": type("FakeExceptions", (), {"ApiException": _FakeApiException})},
    )()

    manifest = {"metadata": {"name": "demo"}}

    runtime._apply("sparkoperator.k8s.io", "v1beta2", "spark", "demo", manifest)

    assert runtime._custom_api.create_calls == [manifest]


def test_delete_application_deletes_custom_object() -> None:
    runtime = object.__new__(KubernetesSparkRuntime)
    runtime._custom_api = _FakeCustomObjectsApi(existing=None)
    runtime._k8s_client = type(
        "FakeK8sClient",
        (),
        {"exceptions": type("FakeExceptions", (), {"ApiException": _FakeApiException})},
    )()

    manifest = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {"name": "demo"},
    }

    runtime.delete_application(manifest, "spark")

    assert runtime._custom_api.delete_calls == [
        {
            "group": "sparkoperator.k8s.io",
            "version": "v1beta2",
            "namespace": "spark",
            "plural": "sparkapplications",
            "name": "demo",
        }
    ]


def test_run_application_reads_driver_logs_and_error_message_for_failed_status() -> None:
    runtime = object.__new__(KubernetesSparkRuntime)
    runtime._custom_api = _FakeCustomObjectsApi(existing=None)
    runtime._core_api = _FakeCoreApi(logs="java.lang.OutOfMemoryError: Java heap space")
    runtime._k8s_client = type(
        "FakeK8sClient",
        (),
        {"exceptions": type("FakeExceptions", (), {"ApiException": _FakeApiException})},
    )()
    runtime._apply = lambda group, version, namespace, name, manifest: None  # type: ignore[method-assign]
    runtime._wait_for_completion = lambda group, version, namespace, name: {  # type: ignore[method-assign]
        "status": {
            "sparkApplicationId": "spark-abc123abc123abc123abc123abc123ab",
            "applicationState": {
                "state": "FAILED",
                "errorMessage": "driver container failed with ExitCode: 1",
            },
        }
    }

    result = runtime.run_application(
        {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {"name": "demo"},
        },
        "spark",
    )

    assert result.app_id == "spark-abc123abc123abc123abc123abc123ab"
    assert result.final_state == "FAILED"
    assert "OutOfMemoryError" in result.driver_logs
    assert "ExitCode: 1" in result.error_message
