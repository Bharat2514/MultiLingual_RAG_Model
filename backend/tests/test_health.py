"""Foundation contract tests: health, error envelope, correlation IDs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_liveness_returns_ok(client):
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["request_id"].startswith("req_")


async def test_response_carries_correlation_headers(client):
    response = await client.get("/api/v1/health")

    assert response.headers["X-Request-ID"].startswith("req_")
    assert float(response.headers["X-Response-Time-ms"]) >= 0


async def test_inbound_request_id_is_honoured(client):
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "req_from_frontend"})

    assert response.headers["X-Request-ID"] == "req_from_frontend"
    assert response.json()["request_id"] == "req_from_frontend"


async def test_unknown_route_uses_the_error_envelope(client):
    response = await client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"request_id", "error"}
    assert set(body["error"]) >= {"code", "message"}
    # A bad route is NOT_FOUND, not DOCUMENT_NOT_FOUND — that code is reserved
    # for a real missing document so the UI can distinguish the two.
    assert body["error"]["code"] == "NOT_FOUND"


async def test_readiness_reports_each_dependency(client):
    response = await client.get("/api/v1/health/ready")

    body = response.json()
    assert {c["name"] for c in body["components"]} == {
        "postgres",
        "redis",
        "qdrant",
        "models",
        "llm",
    }
    assert body["status"] in {"ok", "degraded", "down"}


async def test_root_advertises_the_api_prefix(client):
    body = (await client.get("/")).json()

    assert body["name"] == "PolyDoc"
    assert body["api"] == "/api/v1"
