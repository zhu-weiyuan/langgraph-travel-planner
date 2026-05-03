# LangGraph Travel Planner Agent 🌍✈️

基于 LangGraph + Qwen3.6-27B 的智能旅游规划助手，支持意图识别、目的地推荐、行程规划、预算估算和实时天气查询。

## Demo

```
用户: "你好"
AI:   "嗨！我是你的旅游规划助手 🌍 说说你想去哪？"

用户: "你觉得我应该去哪里玩？"
AI:   "根据你的兴趣，我推荐以下目的地：
       1. 云南 — 春暖花开，适合自然风光爱好者...
       2. 日本京都 — 樱花季绝美体验...
       3. 三亚 — 夏日海滩度假首选..."

用户: "帮我规划去云南5天行程"
AI:   [彩云天气实况] ⛅ 多云 20°C | 湿度 31%
       [逐字流式输出行程规划 + 费用明细 + 天气预警]...
```

## Features

- 🧠 **意图识别** — 自动区分闲聊 / 推荐 / 规划三种意图，不会对所有消息都走完整流程
- 💬 **闲聊模式** — "你好"、"在吗"等问候简短回复 + 引导，不浪费 token
- 🗺️ **目的地推荐** — 用户没有明确目的地时，根据季节/兴趣/预算推荐 3-5 个选择
- 📋 **行程规划** — 详细每日行程（景点/美食/交通/住宿），支持最多 3 轮反馈迭代优化
- 💰 **预算估算** — 按交通/住宿/餐饮/门票/购物分项计算，含省钱建议
- 🌤️ **彩云天气 API** — 实况天气 + 分钟级降雨 + 15 天预报 + 生活指数（穿衣/紫外线/感冒风险）+ 天气预警
- ⚡ **SSE 流式输出** — Server-Sent Events 逐字显示，无需等待完整生成
- 🧹 **Markdown 清理** — 自动去除 `**`、`#`、`|` 等格式符号，输出清爽易读

## Tech Stack

- LangGraph (StateGraph + MemorySaver/SqliteSaver)
- Qwen3.6-27B via llama.cpp (:8080, 本地运行)
- Python Web UI (:7861)
- 彩云天气 API (主) + wttr.in (备用)

## Architecture

```
用户输入
    ↓
classify_intent ──chat──→ chat_reply → END (简短回复)
    ↓ recommend
recommend_destinations → END (推荐3-5个目的地)
    ↓ plan
parse_request → research_destinations ─┐
              → check_weather(彩云) ───┼→ plan_itinerary
                                       ↓
                              should_refine? → refine_plan → plan_itinerary (循环≤3次)
                                       ↓
                              estimate_budget → format_output → END
```

## Quick Start

### 1. 启动本地 LLM

```bash
# llama.cpp 已在 :8080 运行 Qwen3.6-27B
```

### 2. 启动 Web UI

```bash
python app.py
# 访问 http://localhost:7861
```

### 3. 环境变量（可选）

```bash
# 自定义彩云天气 token（默认已内置）
set CAIYUN_TOKEN=your_token_here

# 使用 SQLite 持久化
set USE_SQLITE=1
```

## Project Structure

```
langgraph-travel-planner/
├── agent/
│   ├── state.py            # TravelPlanState (TypedDict)
│   ├── nodes.py            # 意图分类 / 闲聊 / 推荐 / 解析 / 研究 / 天气 / 行程 / 预算 / 输出 / 优化
│   ├── caiyun_weather.py   # 彩云天气 API 集成（实况/小时/天级/分钟降雨/预警/生活指数）
│   └── graph.py            # StateGraph 构建 + 条件路由
├── app.py                  # Web UI + SSE 流式服务端 (:7861)
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/stream` | POST | `{"message": "...", "session_id": "..."}` → SSE 流式响应 |
| `/api/plan` | POST | 同上，返回完整 JSON（兼容旧版） |
| `/api/health` | GET | 健康检查 |

## Intent Routing

| 用户输入 | 意图 | 行为 |
|---------|------|------|
| "你好" / "hi" / "在吗" | `chat` | 简短友好回复 + 引导 |
| "你觉得去哪好？" / "推荐个地方" | `recommend` | 根据季节/兴趣推荐 3-5 个目的地 |
| "帮我规划去云南5天" | `plan` | 完整流程：研究 + 天气 + 行程 + 预算 |

## Weather Integration (彩云天气)

- **实况天气**: 温度 / 湿度 / 气压 / 云量 / 降水 / AQI（1 分钟更新）
- **分钟级降雨**: 未来 2 小时逐分钟，30min/60min 降雨预警
- **小时级预报**: 最多 15 天逐小时
- **天级预报**: 生活指数（穿衣 / 感冒 / 紫外线）
- **天气预警**: 台风 / 暴雨 / 高温等（同步中央气象台）
- **坐标覆盖**: 100+ 国内外城市 + 省级目的地 → 省会坐标
- **限流处理**: 60s 内存缓存 + 429 自动重试
- **备用方案**: wttr.in API

## License

MIT
