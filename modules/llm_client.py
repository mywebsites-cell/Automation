"""Unified LLM + image generation client supporting multiple providers.

Text providers: gemini, openai, anthropic, openai_compatible (DeepSeek/Groq/Ollama/...).
Image providers: openai, openai_compatible, gemini (Imagen).
"""
from __future__ import annotations

import base64
import io
import json
import logging

logger = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": "Gemini (Google)",
    "openai": "OpenAI",
    "anthropic": "Anthropic (Claude)",
    "openai_compatible": "OpenAI-compatible (DeepSeek/Groq/Ollama/OpenRouter...)",
}

IMAGE_PROVIDERS = {
    "openai": "OpenAI (DALL\u00b7E / GPT-Image)",
    "openai_compatible": "OpenAI-compatible",
    "gemini": "Gemini (Imagen)",
}


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _normalize_base_url(base_url: str | None) -> str | None:
    """Strip trailing slashes and any endpoint suffix users may paste by mistake.

    The OpenAI SDK appends `/chat/completions` (and `/images/generations`) itself,
    so a base URL of `.../chat/completions` would otherwise double up.
    """
    if not base_url:
        return None
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    elif url.endswith("/completions"):
        url = url[: -len("/completions")]
    return url or None


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str | None = None, temperature: float = 0.9):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider}")
        if provider == "openai_compatible" and not (base_url or "").strip():
            raise ValueError("OpenAI-compatible provider requires a base URL.")
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.temperature = temperature

    def generate_json(self, prompt: str) -> dict:
        return _parse_json(self._generate_text(prompt))

    def generate_text(self, prompt: str) -> str:
        return self._generate_text(prompt)

    def _generate_text(self, prompt: str) -> str:
        if self.provider == "gemini":
            return self._gemini(prompt)
        if self.provider == "anthropic":
            return self._anthropic(prompt)
        return self._openai(prompt)

    def _gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai is not installed.") from exc

        genai.configure(api_key=self.api_key)
        candidates = [c.strip() for c in self.model.split(",") if c.strip()]
        if not candidates:
            candidates = ["gemini-1.5-flash"]

        last_error: Exception | None = None
        for name in candidates:
            try:
                model = genai.GenerativeModel(name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": self.temperature},
                )
                raw = (response.text or "").strip()
                if not raw:
                    raise ValueError("Gemini returned an empty response")
                return raw
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Gemini generation failed: {last_error}")

    def _openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai is not installed.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise RuntimeError("OpenAI returned an empty response")
        return raw

    def _anthropic(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except Exception as exc:
            raise RuntimeError("anthropic is not installed.") from exc

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not raw:
            raise RuntimeError("Anthropic returned an empty response")
        return raw


class ImageGenerator:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str | None = None):
        if provider not in IMAGE_PROVIDERS:
            raise ValueError(f"Unknown image provider: {provider}")
        if provider == "openai_compatible" and not (base_url or "").strip():
            raise ValueError("OpenAI-compatible image provider requires a base URL.")
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = _normalize_base_url(base_url)

    def generate(self, prompt: str, size: str = "1024x1792") -> bytes:
        if self.provider == "gemini":
            return self._gemini(prompt)
        return self._openai(prompt, size)

    def _openai(self, prompt: str, size: str) -> bytes:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai is not installed.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        response = client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            n=1,
            response_format="b64_json",
        )
        data = response.data[0].b64_json
        if not data:
            raise RuntimeError("Image generation returned no data")
        return base64.b64decode(data)

    def _gemini(self, prompt: str) -> bytes:
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai is not installed.") from exc

        genai.configure(api_key=self.api_key)
        imagen = genai.ImageGenerationModel(self.model)
        result = imagen.generate_images(prompt=prompt, number_of_images=1)
        image = result.images[0]
        pil = getattr(image, "_pil_image", None) or getattr(image, "image", None)
        if pil is None:
            raise RuntimeError("Could not extract a PIL image from Gemini Imagen result")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
