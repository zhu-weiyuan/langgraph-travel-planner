"""
State 定义 - LangGraph Travel Planner Agent

旅游规划助手的状态结构，所有节点通过它传递数据。
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class TravelPlanState(TypedDict):
    # 对话消息历史
    messages: Annotated[List, add_messages]

    # ---- 用户请求解析结果 ----
    destination: Optional[str]          # 目的地（用户指定或推荐）
    destination_country: Optional[str]  # 国家/地区
    days: int                           # 旅行天数
    budget: Optional[float]             # 预算（人民币）
    travelers: int                      # 出行人数
    travel_style: Optional[str]         # 旅行风格: 'budget'/'comfortable'/'luxury'/'adventure'/'cultural'/'beach'
    interests: List[str]                # 兴趣标签: ['美食','摄影','历史','购物','自然']
    season: Optional[str]               # 出行季节/月份
    start_date: Optional[str]           # 出发日期 (YYYY-MM-DD)

    # ---- 研究结果 ----
    destination_info: Optional[str]     # 目的地详细介绍
    attractions: List[str]              # 推荐景点列表
    restaurants: List[str]              # 推荐餐厅/美食
    tips: List[str]                     # 旅行小贴士

    # ---- 天气信息 ----
    weather: Optional[str]              # 目的地天气预报

    # ---- 行程规划 ----
    itinerary: Optional[str]            # 完整行程计划（Markdown格式）
    daily_plans: List[dict]             # 每日详细计划 [{"day":1, "activities":[...], "meals":[...]}]

    # ---- 预算估算 ----
    budget_breakdown: Optional[str]     # 费用明细
    total_estimated_cost: float         # 总费用估算（每人）

    # ---- 输出 ----
    final_output: Optional[str]         # 最终格式化输出

    # ---- 交互控制 ----
    refinement_round: int               # 优化轮次（用户反馈后重新规划）
    user_feedback: Optional[str]        # 用户反馈（用于迭代优化）
    session_id: Optional[str]           # 会话ID
