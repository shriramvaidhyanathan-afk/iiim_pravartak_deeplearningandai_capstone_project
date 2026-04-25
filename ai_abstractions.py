from dataclasses import dataclass
from enum import Enum
from typing import Protocol, List, Optional


class Provider(str, Enum):
    OLLAMA = "ollama"
    GOOGLE = "google"


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    chat_model: str
    embedding_model: str
    temperature: float = 0.0
    keep_alive_seconds: int = 900


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Returns a text completion for the prompt."""


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds multiple texts."""

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query text."""


class ImageCaptioner(Protocol):
    def caption(self, image_bytes: bytes) -> str:
        """Returns a caption for an image."""


@dataclass(frozen=True)
class AIProviderBundle:
    llm: LLMClient
    embeddings: EmbeddingClient
