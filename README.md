# LangGraph Travel Planner Agent 🌍✈️

基于 LangGraph 的智能旅游规划助手。

## 功能
- 🗺️ **目的地推荐** — 根据预算、天数、偏好推荐目的地
- 📋 **行程规划** — 生成详细每日行程（交通/住宿/餐饮/景点）
- 💰 **预算估算** — 自动计算各项费用
- 🌤️ **天气查询** — 集成实时天气信息
- 🎯 **个性化定制** — 根据用户画像调整推荐

## 技术栈
- LangGraph (StateGraph + Checkpointer)
- Qwen3.6-27B (llama.cpp :8080, 本地 LLM)
- Python Web UI (:7861)
- wttr.in API (免费天气)

## 运行
```bash
python app.py
# 访问 http://localhost:7861
```

## 架构
```
用户输入 → parse_request → [research_destinations] → plan_itinerary → estimate_budget → format_output
                                    ↓
                           (optional: check_weather)
                                    ↓
                           refine_based_on_feedback (循环)
```
