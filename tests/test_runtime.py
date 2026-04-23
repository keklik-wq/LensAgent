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
