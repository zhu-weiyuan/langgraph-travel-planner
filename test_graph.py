# -*- coding: utf-8 -*-
"""Travel Planner Agent 测试（不依赖 LLM）"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from agent.state import TravelPlanState
from agent.graph import _build_core_graph, build_graph


def test_state():
    """Test state definition."""
    state = {
        'messages': [],
        'destination': 'Tokyo',
        'days': 5,
        'budget': 8000.0,
        'travelers': 2,
        'travel_style': 'comfortable',
        'interests': ['美食', '购物'],
        'itinerary': None,
        'final_output': None,
        'refinement_round': 0,
        'user_feedback': None,
        'session_id': 'test-123',
    }
    print("✅ State definition OK")


def test_graph_structure():
    """Test graph topology."""
    graph = _build_core_graph()

    nodes = list(graph.nodes.keys())
    expected = ['parse_request', 'research_destinations', 'check_weather',
                'plan_itinerary', 'estimate_budget', 'format_output', 'refine_plan']

    for node in expected:
        assert node in nodes, f"Missing node: {node}"

    print(f"✅ Graph structure OK — Nodes: {nodes}")


def test_compiled_graph():
    """Test compiled graph with checkpointer."""
    graph = build_graph()
    assert graph is not None
    print("✅ Compiled graph OK")


if __name__ == "__main__":
    test_state()
    test_graph_structure()
    test_compiled_graph()
    print("\n🎉 All tests passed!")
