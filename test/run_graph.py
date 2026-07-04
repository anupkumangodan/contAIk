from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.core.workflow import build_parallel_contaik_agent

agent = build_parallel_contaik_agent()

result = agent.invoke({
    "user_query": "write a blog about restaurants in San Jose",
    "messages": [],
    "tasks": [],
    "requires_synthesis": False,
    "agent_results": [],
    "final_answer": "",
    "research_messages": [],
    "blogger_messages": [],
})

print(result["final_answer"])
