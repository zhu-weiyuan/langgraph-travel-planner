"""
节点函数 — LangGraph Travel Planner Agent

流程：
1. classify_intent  — 意图分类（闲聊 / 推荐 / 旅行规划）
2. chat_reply       — 闲聊回复
3. recommend_destinations — 目的地推荐
4. parse_request    — 解析用户需求（目的地/天数/预算/风格/兴趣）
5. research_destinations  — 研究目的地信息（景点/美食/贴士）
6. check_weather    — 查询天气
7. plan_itinerary   — 生成详细行程
8. estimate_budget  — 估算费用
9. format_output    — 格式化最终输出
10. refine_plan     — 根据反馈迭代优化

所有 LLM 调用通过注入的 :class:`LLMClient` 完成，
硬编码的关键词 / 映射表已迁移到 ``config`` 包。
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
import sys
import io
import urllib.request
import json
import re
import random
import datetime
import logging

# Windows console UTF-8 fix (only if not already redirected)
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (OSError, ValueError):
        pass  # LangGraph may have already closed/redirected stdout

# ---- Config imports (centralized data) ----
from config.destinations import CITY_EN_MAP, KNOWN_PLACES
from config.intent_patterns import (
    GREETINGS,
    GREETING_RESPONSES,
    RECOMMEND_KEYWORDS,
    DESTINATION_INDICATORS,
    INTEREST_KEYWORDS,
    BUDGET_KEYWORDS,
    SEASON_KEYWORDS,
    STYLE_KEYWORDS,
    TRAVEL_STYLE_MAP,
)

# ---- LLM Client (injectable) ----
from .llm_client import LLMClient, LocalLLMClient

# Module-level client — lazily initialised, swap with set_llm_client() for tests
_llm_client: LLMClient = None


def get_llm_client() -> LLMClient:
    """Return the shared :class:`LLMClient` singleton (created on first call)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


def set_llm_client(client: LLMClient) -> None:
    """Replace the module-level LLM client (useful for testing / custom backends)."""
    global _llm_client
    _llm_client = client


# Safe print that works even when stdout is closed
def _safe_print(msg):
    """Print safely in multi-threaded LangGraph context."""
    try:
        print(msg)
    except (ValueError, OSError):
        logging.warning(msg)


# ============================================================
# 节点0: 意图分类（闲聊 vs 旅行规划）
# ============================================================

INTENT_SYSTEM = """你是一个意图分类器。判断用户的消息属于哪种类型。

返回严格的 JSON：
{
    "intent": "chat" 或 "recommend" 或 "plan",
    "reason": "简短原因"
}

三种意图：
- "chat"：纯问候/闲聊/感谢/道别（你好、hi、在吗、谢谢、再见）
- "recommend"：用户没有明确目的地，想要推荐（"去哪里好"、"推荐个地方"、"你觉得去哪"、"有什么好去处"）
- "plan"：用户已指定目的地或给出具体旅行需求（"去云南"、"东京5天"、"帮我规划巴黎行程"）

关键区分：
- 有明确目的地 → plan
- 没有目的地但要推荐 → recommend
- 纯聊天 → chat
- 不确定时偏向 recommend（不要默认假设目的地）

不要添加额外文字，只返回 JSON。"""


def classify_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """意图分类节点：区分闲聊 / 推荐 / 规划。"""
    messages = state.get('messages', [])
    user_message = ''
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {'intent': 'chat'}

    # Fast path: common greeting patterns (no LLM needed)
    clean_msg = user_message.strip().rstrip('！？!?.。').strip()
    if clean_msg in GREETINGS or len(clean_msg) <= 2:
        _safe_print(f"[意图分类] chat（快速匹配）: '{user_message}'")
        return {'intent': 'chat'}

    # Fast path: recommend patterns (no LLM needed)
    has_recom_keyword = any(kw in clean_msg for kw in RECOMMEND_KEYWORDS)
    has_dest_indicator = any(kw in clean_msg for kw in DESTINATION_INDICATORS)

    if has_recom_keyword and not has_dest_indicator:
        _safe_print(f"[意图分类] recommend（快速匹配）: '{user_message}'")
        return {'intent': 'recommend', 'user_message': user_message}

    # LLM classification for ambiguous cases
    result = get_llm_client().chat_json(
        [{"role": "user", "content": f"用户消息：{user_message}"}],
        INTENT_SYSTEM,
        max_tokens=64
    )

    intent = result.get('intent', 'recommend')  # default to recommend if unclear
    _safe_print(f"[意图分类] {intent}: '{user_message[:50]}'")
    return {'intent': intent, 'user_message': user_message}


# ============================================================
# 节点1: 闲聊回复
# ============================================================

CHAT_SYSTEM = """你是一个友好的旅游规划助手。用户跟你打招呼或闲聊，请用简短、亲切的方式回应。

规则：
- 简短回复，1-2句话即可
- 主动引导用户说出旅行需求
- 可以提几个例子激发灵感
- 语气轻松自然，不要太正式"""


def chat_reply(state: Dict[str, Any]) -> Dict[str, Any]:
    """闲聊回复节点：简短回应 + 引导。"""
    messages = state.get('messages', [])
    user_message = ''
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    clean_msg = user_message.strip().rstrip('！？!?.。').strip()

    # Check exact or partial match against canned responses
    for key, responses in GREETING_RESPONSES.items():
        if key in clean_msg or clean_msg in key:
            reply = random.choice(responses)
            ai_message = AIMessage(content=reply)
            _safe_print(f"[闲聊回复] '{user_message}' → '{reply[:30]}...'")
            return {'messages': [ai_message], 'final_output': reply}

    # LLM-generated chat reply for other casual messages
    reply = get_llm_client().chat(
        [{"role": "user", "content": user_message}],
        CHAT_SYSTEM,
        max_tokens=128,
        temperature=0.8
    )
    ai_message = AIMessage(content=reply)
    _safe_print(f"[闲聊回复] LLM: '{user_message[:30]}'")
    return {'messages': [ai_message], 'final_output': reply}


# ============================================================
# 节点1b: 目的地推荐（当用户没有明确目的地时）
# ============================================================

RECOMMEND_SYSTEM = """你是一个旅游推荐专家。根据用户的需求和偏好，推荐3-5个适合的旅行目的地。

对每个目的地提供：
1. 目的地名称 + 一句话亮点
2. 适合的季节/月份
3. 大致预算范围（每人）
4. 为什么推荐给用户（结合用户兴趣）

最后让用户选择感兴趣的目的地，你会为他做详细规划。

语气轻松友好，像朋友给建议一样。不要直接生成完整行程——先推荐，等用户选了再做规划。"""


def recommend_destinations(state: Dict[str, Any]) -> Dict[str, Any]:
    """目的地推荐节点：当用户没有明确目的地时，推荐几个选择。"""
    user_message = state.get('user_message', '')

    # Extract interests from config-driven keyword matching
    msg_lower = user_message.lower()
    interests = []
    for category, keywords in INTEREST_KEYWORDS.items():
        if any(w in msg_lower for w in keywords):
            interests.append(category)

    # Detect budget preference
    budget_hint = None
    for budget_type, keywords in BUDGET_KEYWORDS.items():
        if any(w in msg_lower for w in keywords):
            budget_hint = '经济实惠' if budget_type == 'economic' else '高端奢华'
            break

    # Detect season (default to current)
    now = datetime.datetime.now()
    month = now.month
    if month in [3, 4, 5]:
        current_season = '春季（3-5月）'
    elif month in [6, 7, 8]:
        current_season = '夏季（6-8月）'
    elif month in [9, 10, 11]:
        current_season = '秋季（9-11月）'
    else:
        current_season = '冬季（12-2月）'

    # Override with user-specified season keyword
    for kw, s in SEASON_KEYWORDS.items():
        if kw in user_message:
            current_season = s
            break

    interest_str = '、'.join(interests) if interests else '无特定偏好'
    budget_str = f'，预算偏好：{budget_hint}' if budget_hint else ''

    prompt = (f"用户说：\"{user_message}\"\n"
              f"- 当前季节：{current_season}\n"
              f"- 兴趣偏好：{interest_str}{budget_str}\n\n"
              f"请推荐3-5个适合他的旅行目的地。")

    recommendation = get_llm_client().chat(
        [{"role": "user", "content": prompt}],
        RECOMMEND_SYSTEM,
        max_tokens=768,
        temperature=0.9
    )

    ai_message = AIMessage(content=recommendation)
    _safe_print(f"[目的地推荐] 已生成推荐（兴趣={interest_str}，季节={current_season}）")
    return {'messages': [ai_message], 'final_output': recommendation}


# ============================================================
# 节点2: 解析用户请求
# ============================================================

PARSE_SYSTEM = """你是一个旅游需求分析专家。从用户输入中提取关键信息，返回严格的 JSON 格式：

{
    "destination": "目的地（如果用户指定了；否则为null）",
    "destination_country": "国家/地区",
    "days": 天数(整数, 默认3),
    "budget": 预算金额(人民币, 整数或null),
    "travelers": 人数(整数, 默认1),
    "travel_style": "风格(budget/comfortable/luxury/adventure/cultural/beach, 默认comfortable)",
    "interests": ["兴趣标签列表，如美食/摄影/历史/购物/自然/亲子"],
    "season": "出行季节或月份(如'3月','夏天','冬季')",
    "start_date": "出发日期(YYYY-MM-DD格式, 或null)"
}

规则：
- 如果用户没有指定目的地，destination 设为 null，我会推荐
- 从上下文推断旅行风格（穷游→budget，奢华→luxury，探险→adventure）
- 提取所有提到的兴趣点
- 保持简洁，不要添加额外文字"""


def _fallback_parse(user_message: str) -> dict:
    """当 LLM JSON 解析失败时，用正则 + 配置数据提取关键信息。"""
    result = {
        'destination': None,
        'destination_country': None,
        'days': 3,
        'budget': None,
        'travelers': 1,
        'travel_style': 'comfortable',
        'interests': [],
        'season': None,
        'start_date': None,
    }

    # Extract days: "5天", "三天", "3 nights"
    day_match = re.search(r'(\d+)\s*天', user_message)
    if day_match:
        result['days'] = int(day_match.group(1))

    # Extract budget: "预算5000", "¥5000", "5000元"
    budget_match = re.search(r'[预]?[算]?(\d+[千]?)\s*元', user_message)
    if not budget_match:
        budget_match = re.search(r'¥(\d+)', user_message)
    if budget_match:
        val = budget_match.group(1).replace('千', '000')
        result['budget'] = float(val)

    # Extract travelers: "两个人", "一家三口"
    traveler_match = re.search(r'(\d+)\s*个?人', user_message)
    if traveler_match:
        result['travelers'] = int(traveler_match.group(1))

    # Extract destination from common patterns
    dest_patterns = [
        r'去([\u4e00-\u9fff]+[省市区县]?)[的游玩旅行游]',
        r'[到去]([\u4e00-\u9fff]{2,6})[玩旅游度假]',
        r'([\u4e00-\u9fff]{2,6})[的行程规划旅游]',
    ]
    for pattern in dest_patterns:
        m = re.search(pattern, user_message)
        if m:
            result['destination'] = m.group(1)
            break

    # Fuzzy-match against KNOWN_PLACES from config
    if not result['destination']:
        chinese_places = re.findall(r'[\u4e00-\u9fff]{2,6}', user_message)
        for place in chinese_places:
            if place in KNOWN_PLACES:
                result['destination'] = place
                break

    # Detect travel style using config-driven keywords
    for style, keywords in STYLE_KEYWORDS.items():
        if any(w in user_message for w in keywords):
            result['travel_style'] = style
            break

    return result


def parse_request(state: Dict[str, Any]) -> Dict[str, Any]:
    """解析用户需求节点。支持 LLM JSON + regex fallback。"""
    messages = state.get('messages', [])
    user_message = ''
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {
            'destination': None, 'days': 3, 'travelers': 1,
            'travel_style': 'comfortable', 'interests': []
        }

    result = get_llm_client().chat_json(
        [{"role": "user", "content": f"用户需求：{user_message}"}],
        PARSE_SYSTEM
    )

    # Fallback: if LLM didn't return useful JSON, use regex parsing
    if not result.get('destination') and re.search(r'[\u4e00-\u9fff]{2,6}', user_message):
        _safe_print(f"[需求解析] LLM JSON 返回空，使用正则 fallback")
        result = _fallback_parse(user_message)

    # Ensure days is at least what user asked for
    day_check = re.search(r'(\d+)\s*天', user_message)
    if day_check and (not result.get('days') or result['days'] < int(day_check.group(1))):
        result['days'] = int(day_check.group(1))

    # Ensure budget is captured
    budget_check = re.search(r'[预]?[算]?(\d+[千]?)\s*元', user_message) or re.search(r'¥(\d+)', user_message)
    if budget_check and not result.get('budget'):
        val = budget_check.group(1).replace('千', '000')
        result['budget'] = float(val)

    _safe_print(f"[需求解析] 目的地={result.get('destination')}, 天数={result.get('days')}, "
          f"预算={result.get('budget')}, 风格={result.get('travel_style')}")

    return {
        'destination': result.get('destination'),
        'destination_country': result.get('destination_country'),
        'days': int(result.get('days', 3)),
        'budget': float(result.get('budget')) if result.get('budget') else None,
        'travelers': int(result.get('travelers', 1)),
        'travel_style': result.get('travel_style', 'comfortable'),
        'interests': result.get('interests', []),
        'season': result.get('season'),
        'start_date': result.get('start_date'),
    }


# ============================================================
# 节点2: 研究目的地
# ============================================================

RESEARCH_SYSTEM = """你是一个资深旅游规划师。根据以下信息，为用户提供详细的目的地介绍、推荐景点、美食和旅行贴士。

请用中文回复，格式如下：

## 📍 目的地概况
（简要介绍这个地方的特色、最佳旅游时间、文化背景）

## 🏛️ 必去景点（列出5-8个）
1. **景点名** — 简介 + 建议游玩时长
2. ...

## 🍜 美食推荐（列出4-6家）
1. **餐厅/小吃名** — 特色菜 + 人均价格
2. ...

## 💡 旅行贴士（5条实用建议）
- 交通方式建议
- 最佳游览路线
- 注意事项
- 省钱技巧
- 当地文化礼仪

注意：
- 根据用户的旅行风格调整推荐（budget推荐平价，luxury推荐高端）
- 考虑用户的兴趣标签（美食爱好者多推餐厅，摄影爱好者推荐拍照点）
- 信息要实用、具体、可操作"""


def research_destinations(state: Dict[str, Any]) -> Dict[str, Any]:
    """研究目的地信息节点 — 使用 RAG 检索真实旅游攻略数据。"""
    destination = state.get('destination') or '待推荐'
    days = state.get('days', 3)
    style = state.get('travel_style', 'comfortable')
    interests = state.get('interests', [])
    season = state.get('season', '')
    session_id = state.get('session_id', '')

    interest_str = '、'.join(interests) if interests else '无特定偏好'
    season_str = f"，出行时间：{season}" if season else ""

    # --- RAG: retrieve real travel guide data ---
    rag_context = ""
    try:
        from .travel_rag import build_context as _rag_build_context
        rag_context = _rag_build_context(destination, max_length=1500)
        if rag_context:
            sections = rag_context.count('###')
            print(f"[Travel RAG] {destination}: {sections} 条相关知识")
    except Exception as e:
        print(f"[Travel RAG] 检索失败: {e}")

    # Build system prompt with RAG context + user preferences
    research_system = RESEARCH_SYSTEM
    if rag_context:
        research_system += f"\n\n{rag_context}\n\n以上为真实旅游攻略数据，请优先使用这些信息回答。"

    # Inject user preferences
    if session_id:
        try:
            from .preferences import build_preference_context as _pref_ctx
            pref_context = _pref_ctx(session_id)
            if pref_context:
                research_system += pref_context
        except Exception:
            pass

    prompt = (f"请为以下旅行需求提供详细规划信息：\n"
              f"- 目的地：{destination}\n"
              f"- 天数：{days}天\n"
              f"- 风格：{TRAVEL_STYLE_MAP.get(style, style)}\n"
              f"- 兴趣：{interest_str}{season_str}")

    info = get_llm_client().chat(
        [{"role": "user", "content": prompt}],
        research_system,
        max_tokens=768
    )

    # Extract structured data from the response
    attractions = []
    restaurants = []
    tips = []
    for line in info.split('\n'):
        line = line.strip()
        if line.startswith('**') and '—' in line:
            name = line.split('**')[1].split('**')[0] if '**' in line else line
            if '## 🏛️' not in line and '## 🍜' not in line and '## 💡' not in line and '## 📍' not in line:
                if name:
                    attractions.append(name)

    _safe_print(f"[目的地研究] {destination} — 已生成详细规划信息")
    return {'destination_info': info, 'attractions': attractions[:8], 'restaurants': restaurants[:6], 'tips': tips[:5]}


# ============================================================
# 节点3: 查询天气
# ============================================================

def check_weather(state: Dict[str, Any]) -> Dict[str, Any]:
    """查询目的地天气（彩云天气 API，wttr.in 备用）。"""
    destination = state.get('destination')
    if not destination:
        return {'weather': '暂无目的地信息'}

    # 1. 优先使用彩云天气 API
    try:
        from .caiyun_weather import format_weather_summary
        weather_info = format_weather_summary(destination)
        if weather_info and len(weather_info.strip()) > 20:
            print(f"[天气查询] 彩云天气 {destination} — 成功")
            return {'weather': weather_info}
    except Exception as e:
        print(f"[彩云天气失败] {e}，切换到 wttr.in 备用")

    # 2. 备用：wttr.in  (CITY_EN_MAP from config.destinations)
    city_en = CITY_EN_MAP.get(destination, destination)

    try:
        url = f"https://wttr.in/{city_en}?format=%C+%t+%h+%w&lang=zh"
        req = urllib.request.Request(url, headers={'User-Agent': 'TravelPlanner/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            weather_text = resp.read().decode('utf-8')

        forecast_url = f"https://wttr.in/{city_en}?format=3&lang=zh"
        req2 = urllib.request.Request(forecast_url, headers={'User-Agent': 'TravelPlanner/1.0'})
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            forecast = resp2.read().decode('utf-8')

        weather_info = f"🌤️ {destination} 天气（wttr.in）：\n当前：{weather_text}\n未来3天：\n{forecast}"
        print(f"[天气查询] wttr.in 备用 {destination}")
        return {'weather': weather_info}
    except Exception as e:
        print(f"[天气查询失败] {e}")
        return {'weather': f'⚠️ 暂时无法获取{destination}的天气信息'}


# ============================================================
# 节点4: 生成行程
# ============================================================

ITINERARY_SYSTEM = """你是一个专业旅行规划师。根据以下信息，制定详细的每日行程安排。

输出格式要求：

## 🗓️ {目的地} {天数}天{风格}行程

### Day 1 — {日期/主题}
**上午：**
- 09:00 [活动描述]
- 11:00 [活动描述]

**午餐：** [餐厅推荐]

**下午：**
- 14:00 [活动描述]
- 16:30 [活动描述]

**晚餐：** [餐厅推荐]

**住宿：** [酒店/区域建议]

### Day 2 — ...
（依次类推）

## 🚗 交通建议
（当地交通方式、推荐路线）

## 💰 预算概览
- 交通：¥XXX
- 住宿：¥XXX/晚
- 餐饮：¥XXX/天
- 门票：¥XXX
- 其他：¥XXX
- **总计约：¥XXX/人**

注意：
- 行程要合理，不要一天塞太多活动
- 考虑景点之间的地理位置，合理安排路线
- 根据旅行风格调整（budget推荐免费景点和公交，luxury推荐包车和高端酒店）
- 包含用户感兴趣的元素"""


def plan_itinerary(state: Dict[str, Any]) -> Dict[str, Any]:
    """生成详细行程节点。"""
    destination = state.get('destination') or '待定'
    days = state.get('days', 3)
    style = state.get('travel_style', 'comfortable')
    interests = state.get('interests', [])
    budget = state.get('budget')
    info = state.get('destination_info', '')
    weather = state.get('weather', '')

    interest_str = '、'.join(interests) if interests else '无特定偏好'
    budget_str = f"\n- 预算：¥{budget}/人" if budget else ""

    prompt = (f"请为以下旅行制定详细行程：\n"
              f"- 目的地：{destination}\n"
              f"- 天数：{days}天\n"
              f"- 风格：{TRAVEL_STYLE_MAP.get(style, style)}\n"
              f"- 兴趣：{interest_str}{budget_str}\n\n"
              f"参考信息：\n{info[:1000]}\n\n"
              f"天气信息：\n{weather}")

    itinerary = get_llm_client().chat(
        [{"role": "user", "content": prompt}],
        ITINERARY_SYSTEM,
        max_tokens=1024
    )

    _safe_print(f"[行程规划] {destination} {days}天 — 已生成详细行程")
    return {'itinerary': itinerary}


# ============================================================
# 节点5: 估算预算
# ============================================================

BUDGET_SYSTEM = """你是一个旅游费用估算专家。根据以下行程信息，给出详细的费用估算。

输出格式：

## 💰 费用明细（每人）

| 项目 | 详细说明 | 金额（¥） |
|------|---------|----------|
| ✈️ 交通 | ... | XXX |
| 🏨 住宿 | .../晚 × X晚 | XXX |
| 🍽️ 餐饮 | .../天 × X天 | XXX |
| 🎫 门票 | ... | XXX |
| 🚕 当地交通 | ... | XXX |
| 🛍️ 购物/其他 | ... | XXX |
| **合计** | | **¥XXX** |

预算评级：{'budget': '💚 经济实惠', 'comfortable': '💙 舒适适中', 'luxury': '💎 高端奢华'}[风格]

省钱建议：
1. ...
2. ..."""


def estimate_budget(state: Dict[str, Any]) -> Dict[str, Any]:
    """估算旅行费用节点。"""
    destination = state.get('destination') or '待定'
    days = state.get('days', 3)
    style = state.get('travel_style', 'comfortable')
    budget_limit = state.get('budget')
    itinerary = state.get('itinerary', '')
    travelers = state.get('travelers', 1)

    prompt = (f"请估算以下旅行的费用：\n"
              f"- 目的地：{destination}\n"
              f"- 天数：{days}天\n"
              f"- 人数：{travelers}人\n"
              f"- 风格：{style}"
              + (f"\n- 预算上限：¥{budget_limit}/人" if budget_limit else "")
              + f"\n\n行程参考：\n{itinerary[:800]}")

    breakdown = get_llm_client().chat(
        [{"role": "user", "content": prompt}],
        BUDGET_SYSTEM,
        max_tokens=512
    )

    _safe_print(f"[预算估算] {destination} — 已生成费用明细")
    return {'budget_breakdown': breakdown}


# ============================================================
# 节点6: 格式化输出
# ============================================================

def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """整合所有信息，生成最终输出。"""
    destination = state.get('destination') or '待定'
    days = state.get('days', 3)
    weather = state.get('weather', '')
    itinerary = state.get('itinerary', '')
    budget = state.get('budget_breakdown', '')

    # Combine all sections — only include non-empty, meaningful content
    output_parts = []

    # Weather: only show if it has real data (not fallback messages)
    if weather and '暂无目的地' not in weather and '无法获取' not in weather:
        output_parts.append(weather)
        output_parts.append("")

    if itinerary:
        output_parts.append(itinerary)
        output_parts.append("")

    if budget:
        output_parts.append(budget)
        output_parts.append("")

    # Add a friendly closing
    output_parts.append(f"\n---\n*祝您{destination}之旅愉快！🎉 如有需要调整的地方，随时告诉我。*")

    final = '\n'.join(output_parts)
    ai_message = AIMessage(content=final)

    _safe_print(f"[最终输出] {destination} {days}天行程 — 完成 ({len(final)} chars)")
    return {'messages': [ai_message], 'final_output': final}


# ============================================================
# 节点7: 基于反馈优化（可选循环）
# ============================================================

def refine_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """根据用户反馈优化行程。"""
    feedback = state.get('user_feedback', '')
    itinerary = state.get('itinerary', '')
    destination = state.get('destination') or '待定'

    if not feedback:
        return {}

    refine_system = f"""你是一个旅游规划师。用户对你的行程有以下反馈，请根据反馈优化行程：

原始行程：
{itinerary[:1500]}

用户反馈：{feedback}

请输出优化后的完整行程（保持原有格式）。"""

    refined = get_llm_client().chat(
        [{"role": "user", "content": f"请优化{destination}的行程"}],
        refine_system,
        max_tokens=1024
    )

    _safe_print(f"[行程优化] 第 {state.get('refinement_round', 1)} 轮 — 已根据反馈调整")
    return {'itinerary': refined, 'refinement_round': state.get('refinement_round', 0) + 1}
