from __future__ import annotations

import logging
import operator
import os
from logging.handlers import RotatingFileHandler
from typing import Annotated, List, Literal, TypedDict

from pathlib import Path
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Send
from pydantic import BaseModel, Field

# Chat Model

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    log_dir = Path(os.getenv("CONTAIK_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "router.log"

    handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    ))

    logger.addHandler(handler)
    logger.setLevel(os.getenv("CONTAIK_LOG_LEVEL", "DEBUG").upper())
    logger.propagate = False
    return logger


logger = _setup_logger()


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

llm = ChatOpenAI(
    model=os.getenv("ROUTER_MODEL", os.getenv("CONTENT_RESEARCH_MODEL", "gpt-5")),
    # stream_usage=True,
    temperature=0.2,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="...",  # If you prefer to pass api key in directly
    # base_url="...",
    # organization="...",
    # other params...
)

class AgentTask(BaseModel):
    """A single agent invocation requested by the router."""
    source: Literal["research_agent_node", "blogger_agent_node"]
    user_query: str = Field(
        description="Targeted instruction or query to send to that agent"
    )
    focus: str = ""  # What aspect this agent should focus on

class ClassificationResult(BaseModel):
    """Router output - now supports MULTIPLE parallel tasks."""
    tasks: List[AgentTask] = Field(
        description="List of agents to invoke with their targeted queries"
    )
    requires_synthesis: bool = Field(
        default=False,
        description="Whether multiple agents are being used and synthesis is needed"
    )

##### Main Graph State ######

class ContAIKState(TypedDict):
    """Enhanced state for parallel execution."""

    # Conversation history
    messages: Annotated[List[AnyMessage], operator.add]

    # Current user query
    user_query: str

    # Tasks from router (replaces single classification)
    tasks: List[AgentTask]

    # Flag for synthesis
    requires_synthesis: bool

    # Parallel agent results - KEY: uses operator.add for concurrent writes
    agent_results: Annotated[List[dict], operator.add]

    # Final synthesized answer
    final_answer: str

    research_messages: Annotated[List[AnyMessage], operator.add]
    blogger_messages: Annotated[List[AnyMessage], operator.add]

# Worker/Node state for individual agents
class AgentWorkerState(TypedDict):
    """State passed to each parallel worker."""
    task: AgentTask
    messages: List[AnyMessage]
    agent_results: Annotated[List[dict], operator.add]


def classify_query_parallel(state: ContAIKState) -> dict:
    """
    Updated router that can dispatch to MULTIPLE agents.
    """
    user_query = state["user_query"]
    logger.info("Classifying user query: %s", user_query)

    router_llm = llm
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """ You are a Travel Request Router that can dispatch to MULTIPLE agents in parallel.

AVAILABLE AGENTS:
1. research_agent_node - uses internet research to provide
2. blogger_agent_node - generates blog content

ROUTING RULES:

SINGLE AGENT (requires_synthesis = false):
- research a topic ONLY → research_agent_node

MULTIPLE AGENTS (requires_synthesis = true):
- "research information AND write generate blog content" → BOTH agents
- Requests mentioning BOTH booking AND planning → BOTH agents

When dispatching to BOTH agents:
1. Create a focused sub-query for each agent
2. research_agent_node query should focus on searching internet
3. blogger_agent_node query should focus on itinerary/activities/tips

EXAMPLES:

Query: "what are the activies to do on 4th of July in the bay are, CA, USA"
→ tasks: [{"source": "research_agent_node", "user_query": "find activities "}]
→ requires_synthesis: false

Query: "write a blog on long weekend"
→ tasks: [{"source": "blogger_agent_node", "user_query": "generate blog content on things to do on long weekends"}]
→ requires_synthesis: false

Query: "write a blog about things to do in san francisco on a weekend"
→ tasks: [
    {"source": "research_agent_node", "user_query": "Find places to visit in san francisco", "focus": "tourist spots and sight seeing"},
    {"source": "blogger_agent_node", "user_query": "Plan a week-long itinerary for London", "focus": "daily activities and attractions"}
  ]
→ requires_synthesis: true

Query: "i want to write a blog about restaurants in san jose"
→ tasks: [
    {"source": "research_agent_node", "user_query": "Find highly rated restaurants in San Jose", "focus": "restaurant"},
    {"source": "blogger_agent_node", "user_query": "generate blog content about restaurants", "focus": "daily schedule and activities"}
  ]
→ requires_synthesis: true """
        },
        {"role": "user", "content": user_query}
    ])

    logger.info(
        "Router selected %s task(s); requires_synthesis=%s; sources=%s",
        len(result.tasks),
        result.requires_synthesis,
        [task.source for task in result.tasks],
    )

    return {
        "tasks": result.tasks,
        "requires_synthesis": result.requires_synthesis
    }

def dispatch_to_agents(state: ContAIKState):
    """
    Uses Send API to dispatch tasks to agents in PARALLEL.
    This is the key function that enables concurrent execution.
    """
    tasks = state.get("tasks", [])
    logger.info("Dispatching %s task(s) to agents", len(tasks))

    sends = []
    for task in tasks:
        logger.info(
            "Dispatching task to %s with focus=%s query=%s",
            task.source,
            task.focus,
            task.user_query,
        )

        worker_state = {
              "messages": state.get("messages", []),
              "user_query": task.user_query,
              "tasks": state.get("tasks", []),
              "requires_synthesis": state.get("requires_synthesis", False),
              "agent_results": [],  # Each worker starts fresh
              "final_answer": "",
              "research_messages": [],  # Initialize empty for workers
              "blogger_messages": [],  # Initialize empty for workers
        }

        if task.source == "research_agent_node":
            sends.append(Send("research_agent_node", worker_state))  # Your existing node
        elif task.source == "blogger_agent_node":
            sends.append(Send("blogger_agent_node", worker_state))  # Your existing node
        else:
            logger.warning("Skipping unknown task source: %s", task.source)

    return sends

# Tools
from  tools.search import search_tools

# Augment the LLM with tools

research_tools_by_name = {tool.name: tool for tool in search_tools}
search_model_with_tools = llm.bind_tools(search_tools)


from .instructions import research_instructions

def research_agent_node(state: ContAIKState):
    """Search agent with its own message history"""

    # Use search-specific messages
    search_msgs = state.get("research_messages", [])
    logger.info(
        "Research agent running; existing research_messages=%s",
        len(search_msgs),
    )
    logger.debug(
        "Research agent message types=%s",
        [message.__class__.__name__ for message in search_msgs],
    )

    messages = [SystemMessage(content=research_instructions)] + search_msgs

    # If this is first call, add the user query
    if not search_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))

    response = search_model_with_tools.invoke(messages)
    tool_call_count = len(getattr(response, "tool_calls", []) or [])
    logger.info(
        "Research agent response received; tool_calls=%s content_length=%s",
        tool_call_count,
        len(response.content or ""),
    )

    return {
        "research_messages": [response],
        "agent_results": [{
            "agent": "research_agent",
            "focus": "restaurant and blogging",
            "result": response.content
        }] if not response.tool_calls else []
    }

def research_tool_node(state: ContAIKState):
    """Execute search tools"""
    research_msgs = state.get("research_messages", [])
    logger.info("Research tool node running; research_messages=%s", len(research_msgs))

    # Check if research_messages exists and has content
    if not research_msgs:
        logger.warning("Research tool node called without research messages")
        return {"research_messages": []}

    last_message = research_msgs[-1]

    # Check if last_message has tool_calls
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.info("Research tool node found no tool calls")
        return {"research_messages": []}

    result = []
    for tool_call in last_message.tool_calls:
        tool = research_tools_by_name.get(tool_call["name"])
        if tool:
            logger.info("Invoking research tool: %s", tool_call["name"])
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        else:
            logger.warning("No research tool found for tool call: %s", tool_call["name"])

    logger.info("Research tool node completed; observations=%s", len(result))
    return {"research_messages": result}

from .instructions import blogger_instructions
from src.agents.blog_writer import invoke_blog_writer


def _extract_agent_content(response) -> str:
    if isinstance(response, dict):
        messages = response.get("messages", [])
        logger.info("Extracting content from dict response; messages=%s", len(messages))
        if messages:
            last_message = messages[-1]
            return getattr(last_message, "content", str(last_message))
        return str(response)

    return getattr(response, "content", str(response))


def blogger_agent_node(state: ContAIKState):
    """Planner agent with its own message history"""

    # Use planner-specific messages
    blogger_msgs = state.get("blogger_messages", [])
    logger.info(
        "Blogger agent running; existing blogger_messages=%s",
        len(blogger_msgs),
    )

    messages = [SystemMessage(content=blogger_instructions)] + blogger_msgs

    # If this is first call, add the user query
    if not blogger_msgs:
        messages.append(HumanMessage(content=state.get("user_query", "")))

    response = invoke_blog_writer(messages)
    response_content = _extract_agent_content(response)
    response_message = AIMessage(content=response_content)
    logger.info(
        "Blogger agent response extracted; content_length=%s",
        len(response_content),
    )

    return {
        "blogger_messages": [response_message],
        "agent_results": [{
            "agent": "bloggr_agent",
            "focus": "generate blog content",
            "result": response_content
        }]
    }

def should_continue_for_researchAgent(state: ContAIKState):
    search_msgs = state.get("research_messages", [])
    if search_msgs and hasattr(search_msgs[-1], 'tool_calls') and search_msgs[-1].tool_calls:
        logger.info("Research agent continuing to research_tool_node")
        return "research_tool_node"
    logger.info("Research agent routing to synthesizer")
    return "synthesizer"

def should_continue_for_bloggerAgent(state: ContAIKState):
    """Check if blogger needs to continue tool loop"""
    blogger_msgs = state.get("blogger_messages", [])
    if blogger_msgs and hasattr(blogger_msgs[-1], 'tool_calls') and blogger_msgs[-1].tool_calls:
        logger.info("Blogger agent continuing to blogger_agent_node")
        return "blogger_agent_node"
    logger.info("Blogger agent routing to synthesizer")
    return "synthesizer"


def synthesizer_node(state: ContAIKState) -> dict:
    """
    Combines results from parallel agents into a unified response.
    Only called when requires_synthesis is True.
    """

    agent_results = state.get("agent_results", [])
    logger.info("Synthesizer running; agent_results=%s", len(agent_results))

    if not agent_results:
        logger.warning("Synthesizer found no agent results")
        return {
            "final_answer": "I couldn't process your request.",
            "messages": [AIMessage(content="I couldn't process your request.")]
        }

    # If only one result, no synthesis needed
    if len(agent_results) == 1:
        result = agent_results[0]["result"]
        logger.info("Synthesizer returning single agent result; length=%s", len(result))
        return {
            "final_answer": result,
            "messages": [AIMessage(content=result)]
        }

    # Format results for synthesis
    results_formatted = "\n\n" + "="*50 + "\n\n".join([
        f"**{r['agent'].upper()}**\nFocus: {r.get('focus', 'N/A')}\n\n{r['result']}"
        for r in agent_results
    ])

    synthesis_prompt = """You are a Content Response Synthesizer. Combine multiple agent outputs into
a single, well-organized, comprehensive response.

RULES:
1. Organize logically (e.g., restaurants, places, experience)
2. Don't repeat information
3. Highlight key recommendations
4. Note any conflicts or alternatives
5. Create clear sections with headers
6. End with actionable next steps

Original Query: {query}

Agent Results:
{results}

Create a unified, helpful response:"""

    response = llm.invoke([
        HumanMessage(content=synthesis_prompt.format(
            query=state["user_query"],
            results=results_formatted
        ))
    ])
    logger.info("Synthesizer response received; content_length=%s", len(response.content))

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)]
    }
