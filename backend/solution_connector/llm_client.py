# -*- coding: utf-8 -*-
"""LLM 클라이언트 (OpenAI, Claude, Groq)"""
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import streamlit as st

from ..utils.constants import LLM_MAX_RETRIES, LLM_RETRY_DELAY, LLM_MAX_TOKENS, LLM_TEMPERATURE

# API Keys (하드코딩 - 필요시 수정)
_OPENAI_API_KEY = ""
_ANTHROPIC_API_KEY = ""
_GROQ_API_KEY = ""


class BaseLLMClient(ABC):
    """LLM 클라이언트 기본 클래스"""
    MAX_RETRIES = LLM_MAX_RETRIES
    RETRY_DELAY = LLM_RETRY_DELAY

    @abstractmethod
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        pass

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_call(_self, system_prompt: str, user_prompt: str) -> str:
        """캐싱된 LLM 호출 - 동일 프롬프트 재사용 시 API 비용 절감 (1시간 TTL)"""
        return _self._call_api(system_prompt, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                return self._cached_call(system_prompt, user_prompt)
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(self.RETRY_DELAY * attempt)
        raise RuntimeError(f"LLM API 호출 실패 ({retries}회 재시도 후): {str(last_error)}")


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT 클라이언트"""
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude 클라이언트"""
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다: pip install anthropic")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=LLM_TEMPERATURE,
        )
        return response.content[0].text


class GroqClient(BaseLLMClient):
    """Groq 클라이언트"""
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("groq 패키지가 필요합니다: pip install groq")
        self._model = model

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content


def create_llm_client(provider: str, api_key: str, model: Optional[str] = None) -> BaseLLMClient:
    """LLM 클라이언트 팩토리"""
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(api_key=api_key, **({"model": model} if model else {}))
    if provider in ("claude", "anthropic"):
        return ClaudeClient(api_key=api_key, **({"model": model} if model else {}))
    if provider == "groq":
        return GroqClient(api_key=api_key, **({"model": model} if model else {}))
    raise ValueError(f"지원하지 않는 LLM provider: '{provider}'. 지원 목록: 'openai', 'claude', 'groq'")


def load_api_key(provider: str) -> Optional[str]:
    """
    API Key 로드 우선순위: 하드코딩 → Dataiku Secrets → 환경변수 → None
    """
    # 0순위: 하드코딩된 키
    hardcoded = {
        "openai": _OPENAI_API_KEY,
        "claude": _ANTHROPIC_API_KEY,
        "groq": _GROQ_API_KEY,
    }
    key = hardcoded.get(provider, "")
    if key and not key.startswith("여기에") and not key.endswith("REMOVED") and len(key) > 10:
        return key

    key_names = {
        "openai": ["OPENAI_API_KEY", "openai_api_key"],
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "anthropic_api_key"],
        "groq": ["GROQ_API_KEY", "groq_api_key"],
    }
    names = key_names.get(provider, [])

    # 1순위: Dataiku Managed Secrets
    try:
        import dataiku
        client = dataiku.api_client()
        auth_info = client.get_auth_info(with_secrets=True)
        secrets = {s["key"]: s["value"] for s in auth_info.get("secrets", [])}
        for name in names:
            if secrets.get(name):
                return secrets[name]
    except Exception:
        pass

    # 2순위: 환경변수
    for name in names:
        val = os.environ.get(name)
        if val:
            return val

    return None
