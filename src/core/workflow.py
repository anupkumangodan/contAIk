from langgraph.graph import StateGraph, START, END
from .router import ContAIKState, classify_query_parallel, research_agent_node, blogger_agent_node, research_tool_node, synthesizer_node, dispatch_to_agents, should_continue_for_researchAgent, should_continue_for_bloggerAgent

def build_parallel_contaik_agent():
    """Build the graph with parallel execution support."""

    builder = StateGraph(ContAIKState)

    # Add nodes
    builder.add_node("orchestrator", classify_query_parallel)
    builder.add_node("research_agent_node", research_agent_node)
    builder.add_node("blogger_agent_node", blogger_agent_node)
    builder.add_node("research_tool_node", research_tool_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Connect edges
    builder.add_edge(START, "orchestrator")

    # Router dispatches to agents via Send API (parallel execution)
    builder.add_conditional_edges(
        "orchestrator",
        dispatch_to_agents,
        ["research_agent_node", "blogger_agent_node"]
    )

    # research agent tool loop
    builder.add_conditional_edges(
        "research_agent_node",
        should_continue_for_researchAgent,
        ["research_tool_node", "synthesizer"]  # Go to synthesizer instead of END
    )
    builder.add_edge("research_tool_node", "research_agent_node")

    # Blogger agent tool loop
    builder.add_conditional_edges(
        "blogger_agent_node",
        should_continue_for_bloggerAgent,
        ["synthesizer"]  # Go to synthesizer instead of END
    )
    builder.add_edge("blogger_agent_node", "synthesizer")

    # End after synthesis
    builder.add_edge("synthesizer", END)

    return builder.compile()
