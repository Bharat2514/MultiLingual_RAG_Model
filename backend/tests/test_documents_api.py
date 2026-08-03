"""Document endpoints, including the cross-session isolation tests the PRD
requires (NFR-6, §16)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

PDF_BYTES = b"%PDF-1.7\n" + b"0" * 500
DOCX_BYTES = b"PK\x03\x04" + b"\x00" * 30 + b"word/document.xml" + b"\x00" * 200

SESSION_A = "sess_alice"
SESSION_B = "sess_bob"


@pytest.fixture(autouse=True)
def _no_real_ingestion(monkeypatch, tmp_path):
    """Keep uploads from dispatching real work, and isolate object storage."""
    monkeypatch.setattr(
        "app.api.v1.documents._dispatch_ingestion", lambda doc_id, background: "test"
    )
    monkeypatch.setattr("app.core.config.settings.storage.local_dir", tmp_path)


def _upload(client, session_id: str, name: str = "report.pdf", content: bytes = PDF_BYTES):
    return client.post(
        "/api/v1/documents",
        files={"files": (name, content, "application/pdf")},
        headers={"X-Session-ID": session_id},
    )


class TestUpload:
    async def test_accepts_a_pdf(self, client):
        response = await _upload(client, SESSION_A)

        assert response.status_code == 202
        body = response.json()
        assert len(body["accepted"]) == 1
        assert body["accepted"][0]["status"] == "queued"
        assert body["session_id"] == SESSION_A

    async def test_rejects_unsupported_type_without_failing_the_batch(self, client):
        """One bad file must not reject the good ones (FR-1, FR-30)."""
        response = await client.post(
            "/api/v1/documents",
            files=[
                ("files", ("good.pdf", PDF_BYTES, "application/pdf")),
                ("files", ("bad.exe", b"\x7fELF\x00\x00", "application/octet-stream")),
            ],
            headers={"X-Session-ID": SESSION_A},
        )

        body = response.json()
        assert len(body["accepted"]) == 1
        assert len(body["rejected"]) == 1
        assert body["rejected"][0]["code"] == "UNSUPPORTED_FILE_TYPE"

    async def test_identical_upload_is_deduplicated(self, client):
        first = (await _upload(client, SESSION_A)).json()
        second = (await _upload(client, SESSION_A)).json()

        assert second["accepted"][0]["duplicate_of"] == first["accepted"][0]["doc_id"]

    async def test_same_bytes_in_another_session_is_not_a_duplicate(self, client):
        a = (await _upload(client, SESSION_A)).json()
        b = (await _upload(client, SESSION_B)).json()

        assert b["accepted"][0]["doc_id"] != a["accepted"][0]["doc_id"]
        assert b["accepted"][0]["duplicate_of"] is None


class TestListAndGet:
    async def test_lists_only_this_session(self, client):
        await _upload(client, SESSION_A, "alice.pdf")
        await _upload(client, SESSION_B, "bob.pdf", PDF_BYTES + b"different")

        listing = (
            await client.get("/api/v1/documents", headers={"X-Session-ID": SESSION_A})
        ).json()

        assert listing["total"] == 1
        assert listing["documents"][0]["filename"] == "alice.pdf"

    async def test_detail_includes_stage_list(self, client):
        doc_id = (await _upload(client, SESSION_A)).json()["accepted"][0]["doc_id"]

        detail = (
            await client.get(f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_A})
        ).json()

        assert detail["doc_id"] == doc_id
        assert "stages" in detail

    async def test_unknown_document_is_404(self, client):
        response = await client.get(
            "/api/v1/documents/does_not_exist", headers={"X-Session-ID": SESSION_A}
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


class TestSessionIsolation:
    """NFR-6 / R8: a session must never see or touch another session's data."""

    async def test_cannot_read_another_sessions_document(self, client):
        doc_id = (await _upload(client, SESSION_A)).json()["accepted"][0]["doc_id"]

        response = await client.get(
            f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_B}
        )

        # Indistinguishable from "not found" — no existence disclosure.
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    async def test_cannot_delete_another_sessions_document(self, client):
        doc_id = (await _upload(client, SESSION_A)).json()["accepted"][0]["doc_id"]

        response = await client.delete(
            f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_B}
        )
        assert response.status_code == 404

        # ...and it is still there for its real owner.
        owner = await client.get(f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_A})
        assert owner.status_code == 200

    async def test_missing_session_header_gets_a_fresh_scope(self, client):
        await _upload(client, SESSION_A)

        listing = (await client.get("/api/v1/documents")).json()

        assert listing["total"] == 0


class TestDelete:
    async def test_deletes_and_is_then_absent(self, client):
        doc_id = (await _upload(client, SESSION_A)).json()["accepted"][0]["doc_id"]

        response = await client.delete(
            f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_A}
        )
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        follow_up = await client.get(
            f"/api/v1/documents/{doc_id}", headers={"X-Session-ID": SESSION_A}
        )
        assert follow_up.status_code == 404
