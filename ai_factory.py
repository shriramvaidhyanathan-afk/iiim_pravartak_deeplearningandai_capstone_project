import os
from typing import Optional, List

import ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ai_abstractions import (
    AIProviderBundle,
    EmbeddingClient,
    ImageCaptioner,
    LLMClient,
    ModelConfig,
    Provider,
)


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str, temperature: float, keep_alive_seconds: int):
        self._llm = ChatOllama(model=model, temperature=temperature, keep_alive=keep_alive_seconds)

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, keep_alive_seconds: int):
        self._client = OllamaEmbeddings(model=model, keep_alive=str(keep_alive_seconds))

    def embed_documents(self, texts):
        return self._client.embed_documents(texts)

    def embed_query(self, text: str):
        return self._client.embed_query(text)


class OllamaImageCaptioner(ImageCaptioner):
    def caption(self, image_bytes: bytes) -> str:
        response = ollama.generate(
            model="moondream",
            prompt=(
                "Describe this technical chart, diagram, or image in detail for a searchable database. "
                "Focus on labels and data."
            ),
            images=[image_bytes],
            keep_alive="15m",
        )
        return response["response"]


class GoogleLLMClient(LLMClient):
    def __init__(self, model: str, temperature: float, google_api_key: str):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Google provider selected but dependency is missing. "
                "Install langchain-google-genai."
            ) from exc
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=google_api_key,
        )

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content


class GoogleEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, google_api_key: str):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Google provider selected but dependency is missing. "
                "Install langchain-google-genai."
            ) from exc
        self._client = GoogleGenerativeAIEmbeddings(model=model, google_api_key=google_api_key)

    def embed_documents(self, texts):
        return self._client.embed_documents(texts)

    def embed_query(self, text: str):
        return self._client.embed_query(text)


def _get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Google provider selected but GOOGLE_API_KEY is not set. "
            "Set the environment variable before starting the app."
        )
    return api_key


def get_google_flash_chat_models() -> List[str]:
    """
    Returns Google chat model IDs that are Flash variants and support generateContent.
    """
    api_key = _get_google_api_key()
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Google model discovery requires google-genai. "
            "Install langchain-google-genai or google-genai."
        ) from exc

    client = genai.Client(api_key=api_key)
    model_ids: List[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []
        model_name = getattr(model, "name", "")
        if "generateContent" not in actions:
            continue
        if "flash" not in model_name.lower():
            continue
        clean_name = model_name.replace("models/", "")
        model_ids.append(clean_name)

    return sorted(set(model_ids))


def get_google_embedding_models() -> List[str]:
    """
    Returns Google model IDs that support embedContent.
    """
    api_key = _get_google_api_key()
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "Google model discovery requires google-genai. "
            "Install langchain-google-genai or google-genai."
        ) from exc

    client = genai.Client(api_key=api_key)
    model_ids: List[str] = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []
        model_name = getattr(model, "name", "")
        if "embedContent" not in actions:
            continue
        clean_name = model_name.replace("models/", "")
        model_ids.append(clean_name)

    return sorted(set(model_ids))


def build_provider_bundle(config: ModelConfig) -> AIProviderBundle:
    if config.provider == Provider.OLLAMA:
        return AIProviderBundle(
            llm=OllamaLLMClient(
                model=config.chat_model,
                temperature=config.temperature,
                keep_alive_seconds=config.keep_alive_seconds,
            ),
            embeddings=OllamaEmbeddingClient(
                model=config.embedding_model,
                keep_alive_seconds=config.keep_alive_seconds,
            )
        )

    if config.provider == Provider.GOOGLE:
        google_api_key = _get_google_api_key()
        return AIProviderBundle(
            llm=GoogleLLMClient(
                model=config.chat_model,
                temperature=config.temperature,
                google_api_key=google_api_key,
            ),
            embeddings=GoogleEmbeddingClient(
                model=config.embedding_model,
                google_api_key=google_api_key,
            ),
            image_captioner=None,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")