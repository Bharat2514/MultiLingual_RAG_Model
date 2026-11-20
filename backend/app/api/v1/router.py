"""v1 API surface."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import answers, chat, documents, feedback, health, insights, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(insights.router)
api_router.include_router(feedback.router)
api_router.include_router(answers.router)
