from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from core.workflow import build_parallel_contaik_agent


def simple_ascii_from_mermaid(mermaid: str) -> str:
    edges = []
    for line in mermaid.splitlines():
        stripped = line.strip().rstrip(";")
        if " --> " in stripped:
            source, target = stripped.split(" --> ", 1)
        elif " -.-> " in stripped:
            source, target = stripped.split(" -.-> ", 1)
        else:
            continue

        edges.append(f"{source} -> {target}")

    return "\n".join(edges)


# Compile the agent
agent = build_parallel_contaik_agent()

graph = agent.get_graph()
mermaid = graph.draw_mermaid()
output_path = Path(__file__).with_name("graph.mmd")
output_path.write_text(mermaid)

print(f"Graph Mermaid file written to: {output_path}")

try:
    print()
    print("ASCII graph:")
    print()
    print(graph.draw_ascii())
except Exception as exc:
    print()
    print(f"ASCII graph unavailable: {exc}")
    print()
    print("Simple ASCII graph:")
    print()
    print(simple_ascii_from_mermaid(mermaid))
    print()
    print("Mermaid graph:")
    print()
    print(mermaid)
