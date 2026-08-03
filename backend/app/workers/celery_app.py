"""Celery application for asynchronous ingestion (§8.2).

Ingestion is CPU- and IO-bound (PyMuPDF, Tesseract, and later torch), so the
worker runs synchronous code in its own processes rather than sharing the API's
event loop. OCR on a 300-page scan takes minutes; it must never block a request.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import settings
from app.core.logging import configure_logging

INGESTION_QUEUE = "ingestion"

celery_app = Celery(
    "polydoc",
    broker=settings.redis.celery_broker_url,
    backend=settings.redis.celery_result_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # One task at a time per process: OCR and embedding are memory-hungry, and
    # prefetching would let one large document starve the queue.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    # A 300-page scanned PDF at ~30s/100 pages (NFR-2) plus embedding headroom.
    task_soft_time_limit=1800,
    task_time_limit=2100,
    result_expires=86400,
    task_default_queue=INGESTION_QUEUE,
    task_routes={"app.workers.tasks.*": {"queue": INGESTION_QUEUE}},
    broker_connection_retry_on_startup=True,
)


@celery_app.on_after_configure.connect
def _setup_logging(sender, **_kwargs):  # pragma: no cover
    configure_logging()


def is_broker_reachable(timeout: float = 1.0) -> bool:
    """Whether a worker can actually be reached.

    The API uses this to decide between dispatching to Celery and running
    ingestion inline, so `make dev` works without a separate worker process.
    """
    try:
        with celery_app.connection(transport_options={"max_retries": 0}) as connection:
            connection.ensure_connection(max_retries=0, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False
