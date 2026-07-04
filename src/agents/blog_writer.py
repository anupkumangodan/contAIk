from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphRecursionError

from src.core.instructions import blogger_instructions


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        name, value = stripped.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name, value)


_load_dotenv()


MODEL_NAME = os.getenv("BLOG_WRITER_MODEL", os.getenv("CONTENT_RESEARCH_MODEL", "gpt-5"))
MODEL_PROVIDER = os.getenv("BLOG_WRITER_MODEL_PROVIDER", "openai")
TEMPERATURE = float(os.getenv("BLOG_WRITER_TEMPERATURE", "0.4"))
CACHE_TTL_SECONDS = int(os.getenv("BLOG_WRITER_CACHE_TTL_SECONDS", "1209600"))
CACHE_DIR = Path(os.getenv("BLOG_WRITER_CACHE_DIR", ".cache/blog_writer"))
RECURSION_LIMIT = int(os.getenv("BLOG_WRITER_RECURSION_LIMIT", "25"))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_blog_writer_agent():
    """Build the blog writer agent once per process."""
    model = init_chat_model(
        api_key=_require_env("OPENAI_API_KEY"),
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE,
    )

    return create_deep_agent(
        model=model,
        instructions=blogger_instructions,
        tools=[],
        builtin_tools=[],
    )


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message

    if isinstance(message, BaseMessage):
        role = {
            "human": "user",
            "ai": "assistant",
        }.get(message.type, message.type)
        return {
            "role": role,
            "content": message.content,
        }

    return {
        "role": "user",
        "content": str(message),
    }


def _normalize_input(prompt_or_messages: str | list[Any]) -> dict[str, Any]:
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = [_message_to_dict(message) for message in prompt_or_messages]

    return {"messages": messages}


def _cache_key(payload: dict[str, Any]) -> str:
    cache_payload = {
        "payload": payload,
        "instructions": blogger_instructions,
        "model": MODEL_NAME,
        "model_provider": MODEL_PROVIDER,
        "temperature": TEMPERATURE,
    }
    serialized = json.dumps(cache_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_cache(path: Path, ttl_seconds: int) -> Any | None:
    if not path.exists():
        return None

    if ttl_seconds > 0 and time.time() - path.stat().st_mtime > ttl_seconds:
        return None

    with path.open("rb") as cached_file:
        return pickle.load(cached_file)


def _write_cache(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as cached_file:
        pickle.dump(value, cached_file)


def invoke_blog_writer(
    prompt_or_messages: str | list[Any],
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    recursion_limit: int = RECURSION_LIMIT,
) -> Any:
    payload = _normalize_input(prompt_or_messages)
    cache_path = CACHE_DIR / f"{_cache_key(payload)}.pickle"

    if use_cache:
        cached_result = _read_cache(cache_path, cache_ttl_seconds)
        if cached_result is not None:
            return cached_result

    try:
        result = get_blog_writer_agent().invoke(
            payload,
            config={"recursion_limit": recursion_limit},
        )
    except GraphRecursionError as exc:
        raise RuntimeError(
            "The blog writer agent did not reach a final answer before the "
            f"recursion limit ({recursion_limit}). Try a narrower prompt or "
            "increase BLOG_WRITER_RECURSION_LIMIT."
        ) from exc

    if use_cache:
        _write_cache(cache_path, result)

    return result


if __name__ == "__main__":
    result = invoke_blog_writer(
        "Write a short blog post about a winter food and history trip in Delhi."
    )
    print(result)
