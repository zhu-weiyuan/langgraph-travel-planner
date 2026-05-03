"""
Graph builder - LangGraph Travel Planner Agent

流程：
parse_request → research_destinations → check_weather → plan_itinerary → estimate_budget → format_output
                                                            ↓
                                              (用户反馈? refine_plan → plan_itinerary → ...)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import TravelPlanState
from .nodes import (
    parse_request,
    research_destinations,
    check_weather,
    plan_itinerary,
    estimate_budget,
    format_output,
    refine_plan,
)


def should_refine(state: dict) -> str:
    """判断是否需要根据反馈优化行程。"""
    feedback = state.get('user_feedback')
    refinement_round = state.get('refinement_round', 0)

    if feedback and refinement_round < 3:
        return 'refine_plan'
    return 'estimate_budget'


def _build_core_graph():
    """构建 StateGraph 拓扑。"""
    graph = StateGraph(TravelPlanState)

    # Nodes
    graph.add_node('parse_request', parse_request)
    graph.add_node('research_destinations', research_destinations)
    graph.add_node('check_weather', check_weather)
    graph.add_node('plan_itinerary', plan_itinerary)
    graph.add_node('estimate_budget', estimate_budget)
    graph.add_node('format_output', format_output)
    graph.add_node('refine_plan', refine_plan)

    # Entry: parse user request
    graph.add_edge(START, 'parse_request')

    # Research phase (parallel research + weather)
    graph.add_edge('parse_request', 'research_destinations')
    graph.add_edge('parse_request', 'check_weather')

    # Plan itinerary after research completes
    graph.add_edge('research_destinations', 'plan_itinerary')
    graph.add_edge('check_weather', 'plan_itinerary')

    # After planning: check if refinement needed
    graph.add_conditional_edges(
        'plan_itinerary',
        should_refine,
        {'refine_plan': 'refine_plan', 'estimate_budget': 'estimate_budget'}
    )

    # Refinement loop: refine → re-plan → check again
    graph.add_edge('refine_plan', 'plan_itinerary')

    # Budget estimation
    graph.add_edge('estimate_budget', 'format_output')

    # Final output
    graph.add_edge('format_output', END)

    return graph


def build_graph(use_sqlite: bool = False, db_path: str = "travel_checkpoints.db"):
    """构建并编译旅游规划 Agent 图。"""
    graph = _build_core_graph()

    if use_sqlite:
        import sqlite3
        conn = sqlite3.connect(db_path)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        print(f"[Graph] 使用 SQLite checkpointer: {db_path}")
    else:
        checkpointer = MemorySaver()
        print("[Graph] 使用内存 checkpointer（测试模式）")

    compiled = graph.compile(checkpointer=checkpointer)

    print(f"[Graph] Travel Planner Agent 编译完成")
    print(f"[Graph] 节点: {list(graph.nodes.keys())}")
    return compiled


def create_agent():
    """便捷函数。"""
    return build_graph()
