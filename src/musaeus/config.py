"""Configuration — where Musaeus reads its settings from.

One small dataclass, loaded from environment variables (and an optional .env).
Keeping this in one place means every module asks the *same* source for keys and
model names, instead of scattering os.getenv calls across the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader — no dependency. Lines like KEY=value become env vars.

    We do this ourselves (instead of python-dotenv) so a reader can see exactly
    what "loading .env" means: read the file, set any key that isn't already set.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    provider: str = "local"                 # local | anthropic | openai | google
    # Per-provider model + credentials. `local` uses an OpenAI-compatible gateway.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "gemma3"
    # Sane default models per provider (override with --model or MUSAEUS_*_MODEL).
    # `gemini-flash-latest` is Google's moving alias for the current Flash model,
    # so the default keeps working as new generations ship.
    anthropic_model: str = "claude-opus-5"
    openai_model: str = "gpt-5-mini"
    google_model: str = "gemini-flash-latest"

    @property
    def model_for(self) -> str:
        return {
            "local": self.local_model,
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "google": self.google_model,
        }[self.provider]


def load_settings(provider: str | None = None, model: str | None = None) -> Settings:
    _load_dotenv()
    s = Settings(
        provider=provider or os.getenv("MUSAEUS_PROVIDER", "local"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        local_base_url=os.getenv("MUSAEUS_LOCAL_BASE_URL", "http://localhost:11434/v1"),
        local_model=os.getenv("MUSAEUS_LOCAL_MODEL", "gemma3"),
        anthropic_model=os.getenv("MUSAEUS_ANTHROPIC_MODEL", "claude-opus-5"),
        openai_model=os.getenv("MUSAEUS_OPENAI_MODEL", "gpt-5-mini"),
        google_model=os.getenv("MUSAEUS_GOOGLE_MODEL", "gemini-flash-latest"),
    )
    if model:
        # Override whichever provider's model is active.
        field = {
            "local": "local_model",
            "anthropic": "anthropic_model",
            "openai": "openai_model",
            "google": "google_model",
        }[s.provider]
        s = Settings(**{**s.__dict__, field: model})
    return s
