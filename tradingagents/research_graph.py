"""Common analyst/tool, debate and risk graph used by both the platform and the ETF CLI."""
from langgraph.graph import END, START, StateGraph


def build_research_graph(state_type, analysts, nodes, clears, tools, display, routes,
                         team, routing, *, debate=True, analysts_only=False):
    graph = StateGraph(state_type)
    first = f"{display(analysts[0])} Analyst"
    graph.add_edge(START, first)
    for index, role in enumerate(analysts):
        label = display(role)
        analyst, clear = f"{label} Analyst", f"Msg Clear {label}"
        graph.add_node(analyst, nodes[role])
        graph.add_node(clear, clears[role])
        if role in tools:
            tool = f"tools_{role}"
            graph.add_node(tool, tools[role])
            graph.add_conditional_edges(analyst, routes[role], [tool, clear])
            graph.add_edge(tool, analyst)
        else:
            graph.add_edge(analyst, clear)
        if index + 1 < len(analysts):
            target = f"{display(analysts[index + 1])} Analyst"
        elif analysts_only:
            target = END
        else:
            target = "Bull Researcher" if debate else "Research Manager"
        graph.add_edge(clear, target)

    if not analysts_only:
        for name, node in team.items():
            graph.add_node(name, node)
        if debate:
            for speaker, opponent in (("Bull Researcher", "Bear Researcher"), ("Bear Researcher", "Bull Researcher")):
                graph.add_conditional_edges(speaker, routing.should_continue_debate, [opponent, "Research Manager"])
            graph.add_edge("Trader", "Risky Analyst")
            for speaker, following in (("Risky Analyst", "Safe Analyst"), ("Safe Analyst", "Neutral Analyst"),
                                       ("Neutral Analyst", "Risky Analyst")):
                graph.add_conditional_edges(speaker, routing.should_continue_risk_analysis, [following, "Risk Judge"])
        else:
            graph.add_edge("Trader", "Safe Analyst")
            graph.add_edge("Safe Analyst", "Risk Judge")
        graph.add_edge("Research Manager", "Trader")
        graph.add_edge("Risk Judge", END)
    return graph.compile()
