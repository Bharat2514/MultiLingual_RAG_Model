#!/usr/bin/env bash
# Install the macOS system dependencies PolyDoc needs outside Python.
# The containers install these themselves (see backend/Dockerfile); this is for
# running the API and worker on the host so the models get Apple MPS.
set -euo pipefail

info()  { printf '\033[0;34m→\033[0m %s\n' "$1"; }
ok()    { printf '\033[0;32m✓\033[0m %s\n' "$1"; }
warn()  { printf '\033[0;33m!\033[0m %s\n' "$1"; }
fail()  { printf '\033[0;31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || fail "This script is for macOS. On Linux: apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng"

command -v brew >/dev/null 2>&1 || fail "Homebrew not found. Install it from https://brew.sh"

# --- Tesseract + Hindi/English traineddata (FR-4) ----------------------------
if command -v tesseract >/dev/null 2>&1; then
  ok "tesseract already installed ($(tesseract --version 2>&1 | head -1))"
else
  info "Installing tesseract..."
  brew install tesseract
  ok "tesseract installed"
fi

# tesseract-lang ships every language pack, including `hin`.
if tesseract --list-langs 2>/dev/null | grep -qx hin; then
  ok "Hindi (hin) traineddata present"
else
  info "Installing tesseract-lang (adds Hindi)..."
  brew install tesseract-lang
  tesseract --list-langs 2>/dev/null | grep -qx hin \
    && ok "Hindi (hin) traineddata present" \
    || warn "Hindi traineddata still missing — check 'brew --prefix tesseract-lang'"
fi

# --- Devanagari rendering for any generated PDFs/plots (NFR-10) --------------
if brew list --cask font-noto-sans-devanagari >/dev/null 2>&1; then
  ok "Noto Sans Devanagari font present"
else
  info "Installing Noto Sans Devanagari font..."
  brew install --cask font-noto-sans-devanagari || warn "Font install failed (non-fatal)"
fi

# --- Docker ------------------------------------------------------------------
if docker info >/dev/null 2>&1; then
  ok "Docker daemon reachable"
else
  warn "Docker is not running — start Docker Desktop before 'make infra'"
fi

echo
ok "System dependencies ready."
echo "  Next:  make setup && make infra && make dev"
