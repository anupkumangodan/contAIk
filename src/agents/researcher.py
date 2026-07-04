from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.errors import GraphRecursionError


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


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    log_dir = Path(os.getenv("CONTAIK_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "researcher.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    ))

    logger.addHandler(handler)
    logger.setLevel(os.getenv("CONTAIK_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


logger = _setup_logger()


RESEARCH_INSTRUCTIONS = """You are a Deep Research Agent specializing in content creation.

Your primary responsibility is to gather, analyze, and organize high-quality information that will be used by downstream content generation agents. You are not responsible for writing the final article.

Objectives:

* Develop a comprehensive understanding of the assigned topic.
* Identify key concepts, terminology, trends, and supporting evidence.
* Gather information from credible and authoritative sources.
* Distinguish established facts from opinions and speculation.
* Identify conflicting viewpoints and summarize each objectively.
* Note any gaps, uncertainties, or areas requiring further research.
* Maintain factual accuracy and avoid unsupported claims.

For every research task, produce:

1. Executive summary
2. Key concepts and definitions
3. Major findings
4. Supporting facts, statistics, and examples
5. Expert opinions or differing viewpoints
6. Relevant historical or industry context
7. Emerging trends and recent developments
8. Source references
9. Research gaps and confidence level

Guidelines:

* Prioritize authoritative and trustworthy sources.
* Synthesize information rather than copying source material.
* Preserve technical accuracy while remaining clear and organized.
* Clearly indicate when information is uncertain or unavailable.
* Structure findings so they can be easily consumed by downstream agents such as Outliner, Blog Writer, SEO Optimizer, or Fact Checker.
* Keep the research pass focused: use at most one web-search pass unless the user explicitly asks for deeper research.
* Return the final research brief as soon as enough credible source material is available.
"""

MODEL_NAME = os.getenv("CONTENT_RESEARCH_MODEL", "gpt-5")
MODEL_PROVIDER = os.getenv("CONTENT_RESEARCH_MODEL_PROVIDER", "openai")
TEMPERATURE = float(os.getenv("CONTENT_RESEARCH_TEMPERATURE", "0.2"))
CACHE_TTL_SECONDS = int(os.getenv("CONTENT_RESEARCH_CACHE_TTL_SECONDS", "1209600"))
CACHE_DIR = Path(os.getenv("CONTENT_RESEARCH_CACHE_DIR", ".cache/content_research"))
RECURSION_LIMIT = int(os.getenv("CONTENT_RESEARCH_RECURSION_LIMIT", "50"))

SEARCH_CONFIG = {
    "max_results": int(os.getenv("TAVILY_MAX_RESULTS", "5")),
    "topic": os.getenv("TAVILY_TOPIC", "general"),
    "include_images": os.getenv("TAVILY_INCLUDE_IMAGES", "false").lower() == "true",
    "include_image_descriptions": os.getenv(
        "TAVILY_INCLUDE_IMAGE_DESCRIPTIONS", "false"
    ).lower()
    == "true",
    "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def _get_tavily_search() -> TavilySearch:
    _require_env("TAVILY_API_KEY")
    logger.info("Creating TavilySearch tool with config=%s", SEARCH_CONFIG)
    return TavilySearch(**SEARCH_CONFIG)


@tool
def internet_search(query: str) -> str:
    """Search the internet for current or specific research information."""
    logger.info("Calling internet_search tool with query=%s", query)
    result = _get_tavily_search().invoke({"query": query})
    logger.debug("internet_search result=%s", json.dumps(result, default=str))
    return json.dumps(result, default=str)


@lru_cache(maxsize=1)
def get_content_research_agent():
    """Build the agent once per process instead of rebuilding it on every call."""
    logger.info(
        "Creating content research agent model=%s provider=%s temperature=%s",
        MODEL_NAME,
        MODEL_PROVIDER,
        TEMPERATURE,
    )
    model = init_chat_model(
        api_key=_require_env("OPENAI_API_KEY"),
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=TEMPERATURE,
    )

    return create_deep_agent(
        model=model,
        instructions=RESEARCH_INSTRUCTIONS,
        tools=[internet_search],
        builtin_tools=[],
    )


def _normalize_input(prompt_or_messages: str | list[dict[str, str]]) -> dict[str, Any]:
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    return {"messages": messages}


def _cache_key(payload: dict[str, Any]) -> str:
    cache_payload = {
        "payload": payload,
        "instructions": RESEARCH_INSTRUCTIONS,
        "model": MODEL_NAME,
        "model_provider": MODEL_PROVIDER,
        "temperature": TEMPERATURE,
        "search_config": SEARCH_CONFIG,
    }
    serialized = json.dumps(cache_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_cache(path: Path, ttl_seconds: int) -> Any | None:
    if not path.exists():
        logger.debug("Cache file does not exist: %s", path)
        return None

    if ttl_seconds > 0 and time.time() - path.stat().st_mtime > ttl_seconds:
        logger.info("Cache entry expired: %s", path)
        return None

    with path.open("rb") as cached_file:
        return pickle.load(cached_file)


def _write_cache(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as cached_file:
        pickle.dump(value, cached_file)
    logger.info("Wrote research result to cache: %s", path)


def invoke_content_research(
    prompt_or_messages: str | list[dict[str, str]],
    *,
    use_cache: bool = True,
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    recursion_limit: int = RECURSION_LIMIT,
) -> Any:
    payload = _normalize_input(prompt_or_messages)
    cache_key = _cache_key(payload)
    cache_path = CACHE_DIR / f"{cache_key}.pickle"
    logger.info(
        "Invoking content research; cache_key=%s use_cache=%s recursion_limit=%s",
        cache_key,
        use_cache,
        recursion_limit,
    )
    logger.debug("Content research payload=%s", json.dumps(payload, default=str))

    if use_cache:
        cached_result = _read_cache(cache_path, cache_ttl_seconds)
        if cached_result is not None:
            logger.info("Content research result returned from cache: %s", cache_path)
            logger.debug("Cached content research result=%s", repr(cached_result))
            return cached_result
        logger.info("Content research cache miss; actual agent/tool execution may run")
    else:
        logger.info("Content research cache disabled; actual agent/tool execution may run")

    try:
        logger.info("Calling content research agent")
        result = get_content_research_agent().invoke(
            payload,
            config={"recursion_limit": recursion_limit},
        )
        logger.info("Content research agent returned result")
        logger.debug("Content research result=%s", repr(result))
    except GraphRecursionError as exc:
        logger.exception(
            "Content research agent hit recursion limit=%s",
            recursion_limit,
        )
        raise RuntimeError(
            "The content research agent did not reach a final answer before the "
            f"recursion limit ({recursion_limit}). Try a narrower prompt, lower "
            "TAVILY_MAX_RESULTS, or increase CONTENT_RESEARCH_RECURSION_LIMIT."
        ) from exc
    except Exception:
        logger.exception("Content research agent failed")
        raise

    if use_cache:
        _write_cache(cache_path, result)

    return result


if __name__ == "__main__":
    result = invoke_content_research(
        "Plan a 5 day travel itinerary for a food and historical sites lover during winter in Delhi"
    )
    print(result)
