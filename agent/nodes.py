"""
节点函数 - LangGraph Travel Planner Agent

流程：
1. parse_request — 解析用户需求（目的地/天数/预算/风格/兴趣）
2. research_destinations — 研究目的地信息（景点/美食/贴士）
3. check_weather — 查询天气
4. plan_itinerary — 生成详细行程
5. estimate_budget — 估算费用
6. format_output — 格式化最终输出
"""

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
import sys
import io
import urllib.request
import json
import re
import logging

# Windows console UTF-8 fix (only if not already redirected)
if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (OSError, ValueError):
        pass  # LangGraph may have already closed/redirected stdout

# Safe print that works even when stdout is closed
def _safe_print(msg):
    """Print safely in multi-threaded LangGraph context."""
    try:
        print(msg)
    except (ValueError, OSError):
        logging.warning(msg)

LLM_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLM_API_KEY = "your_key_here"


def _call_llm(messages: List[dict], system: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """调用本地 llama.cpp HTTP API。"""
    payload = {
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM 错误] {e}")
        return "抱歉，我暂时无法处理您的请求，请稍后再试。"


def _call_llm_json(messages: List[dict], system: str, max_tokens: int = 256) -> dict:
    """调用 LLM 并解析 JSON 响应。容错性强，支持多种输出格式。"""
    payload = {
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"].strip()

            # Strategy 1: find JSON object in the response (line by line)
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass

            # Strategy 2: find JSON block with regex (handles multi-line)
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            # Strategy 3: try the whole text (maybe it's just JSON without braces on separate lines)
            try:
                cleaned = text.strip().rstrip("`").strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

            print(f"[JSON 解析失败] 原始输出: {text[:200]}")
            return {}
    except Exception as e:
        print(f"[LLM JSON 错误] {e}")
        return {}


# ============================================================
# 节点0: 意图分类（闲聊 vs 旅行规划）
# ============================================================

INTENT_SYSTEM = """你是一个意图分类器。判断用户的消息是否包含旅行规划相关的需求。

返回严格的 JSON：
{
    "intent": "chat" 或 "plan",
    "reason": "简短原因"
}

规则：
- "chat"：问候（你好/hi/hello）、闲聊、感谢、道别、无意义输入
- "plan"：包含目的地、天数、预算、行程、推荐、旅游等关键词，或明确表达旅行意愿
- 如果用户说"帮我规划"、"我想去"、"推荐"、"多少钱"、"几天"→ plan
- 如果用户只说"你好"、"在吗"、"谢谢"、"再见"→ chat
- 不确定时偏向 plan（让用户得到帮助）

不要添加额外文字，只返回 JSON。"""


def classify_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """意图分类节点：区分闲聊和旅行规划请求。"""
    messages = state.get('messages', [])
    user_message = ''
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {'intent': 'chat'}

    # Fast path: common greeting patterns (no LLM needed)
    greetings = ['你好', '您好', '嗨', 'hi', 'hello', '在吗', '在不在',
                 '早上好', '晚上好', '中午好', '下午好', '嘿', 'hey',
                 '谢谢', '感谢', '再见', '拜拜', 'bye', '好的', '嗯',
                 '哦', '哈哈', '嘻嘻', '嗯嗯', 'ok', 'ok的']
    clean_msg = user_message.strip().rstrip('！？!?.。').strip()
    if clean_msg in greetings or len(clean_msg) <= 2:
        _safe_print(f"[意图分类] chat（快速匹配）: '{user_message}'")
        return {'intent': 'chat'}

    # LLM classification for ambiguous cases
    result = _call_llm_json(
        [{"role": "user", "content": f"用户消息：{user_message}"}],
        INTENT_SYSTEM,
        max_tokens=64
    )

    intent = result.get('intent', 'plan')  # default to plan if unclear
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

    # Common greeting responses (fast path)
    greeting_responses = {
        '你好': ['你好呀！👋 想去哪里玩？告诉我目的地和天数，我来帮你规划行程～',
                 '嗨！我是你的旅游规划助手 🌍 说说你想去哪？'],
        '您好': ['您好！😊 有什么旅行计划吗？我可以帮您做详细的行程规划。',
                 '您好呀！想去哪里玩？告诉我目的地和预算就好啦～'],
        'hi': ['Hi there! 👋 Ready to plan your next trip? Tell me where you want to go!',
               'Hey! I can help you plan a trip. Where are you thinking of going?'],
        'hello': ['Hello! 🌍 Looking for travel inspiration? Just tell me your destination!',
                  'Hi! What trip can I help you plan today?'],
        '在吗': ['在的！有什么旅行计划需要帮忙的吗？🗺️',
                 '在呢～想去哪里玩？告诉我目的地和天数就好！'],
        '谢谢': ['不客气！😊 还有其他问题随时问我～',
                 '应该的！祝你旅途愉快 🎉'],
        '再见': ['再见！旅途愉快～ 🌟',
                 '拜拜！下次旅行还找我规划 😄'],
    }

    clean_msg = user_message.strip().rstrip('！？!?.。').strip()

    # Check exact or partial match
    for key, responses in greeting_responses.items():
        if key in clean_msg or clean_msg in key:
            import random
            reply = random.choice(responses)
            ai_message = AIMessage(content=reply)
            _safe_print(f"[闲聊回复] '{user_message}' → '{reply[:30]}...'")
            return {'messages': [ai_message], 'final_output': reply}

    # LLM-generated chat reply for other casual messages
    reply = _call_llm(
        [{"role": "user", "content": user_message}],
        CHAT_SYSTEM,
        max_tokens=128,
        temperature=0.8
    )
    ai_message = AIMessage(content=reply)
    _safe_print(f"[闲聊回复] LLM: '{user_message[:30]}'")
    return {'messages': [ai_message], 'final_output': reply}


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
    """当 LLM JSON 解析失败时，用正则提取关键信息。"""
    import re
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

    # If no destination found but there's a Chinese place name
    if not result['destination']:
        chinese_places = re.findall(r'[\u4e00-\u9fff]{2,6}', user_message)
        known_places = ['云南', '北京', '上海', '广州', '深圳', '成都', '杭州', '西安',
                        '重庆', '南京', '武汉', '长沙', '昆明', '厦门', '青岛', '大连',
                        '三亚', '丽江', '桂林', '东京', '首尔', '曼谷', '新加坡', '巴黎',
                        '伦敦', '纽约', '悉尼', '迪拜', '欧洲', '日本', '泰国', '马尔代夫']
        for place in chinese_places:
            if place in known_places:
                result['destination'] = place
                break

    # Detect style keywords
    if any(w in user_message for w in ['穷游', '省钱', '经济']):
        result['travel_style'] = 'budget'
    elif any(w in user_message for w in ['奢华', '高端', '豪华']):
        result['travel_style'] = 'luxury'
    elif any(w in user_message for w in ['探险', '冒险', '户外']):
        result['travel_style'] = 'adventure'
    elif any(w in user_message for w in ['文化', '历史', '博物馆']):
        result['travel_style'] = 'cultural'
    elif any(w in user_message for w in ['海边', '海滩', '海岛', '度假']):
        result['travel_style'] = 'beach'

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

    result = _call_llm_json(
        [{"role": "user", "content": f"用户需求：{user_message}"}],
        PARSE_SYSTEM
    )

    # Fallback: if LLM didn't return useful JSON, use regex parsing
    if not result.get('destination') and re.search(r'[\u4e00-\u9fff]{2,6}', user_message):
        _safe_print(f"[需求解析] LLM JSON 返回空，使用正则 fallback")
        result = _fallback_parse(user_message)

    # Ensure days is at least what user asked for
    import re as _re
    day_check = _re.search(r'(\d+)\s*天', user_message)
    if day_check and (not result.get('days') or result['days'] < int(day_check.group(1))):
        result['days'] = int(day_check.group(1))

    # Ensure budget is captured
    budget_check = _re.search(r'[预]?[算]?(\d+[千]?)\s*元', user_message) or _re.search(r'¥(\d+)', user_message)
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
    """研究目的地信息节点。"""
    destination = state.get('destination') or '待推荐'
    days = state.get('days', 3)
    style = state.get('travel_style', 'comfortable')
    interests = state.get('interests', [])
    season = state.get('season', '')

    style_map = {
        'budget': '经济实惠型',
        'comfortable': '舒适休闲型',
        'luxury': '高端奢华型',
        'adventure': '探险刺激型',
        'cultural': '文化深度游',
        'beach': '海滨度假型',
    }

    interest_str = '、'.join(interests) if interests else '无特定偏好'
    season_str = f"，出行时间：{season}" if season else ""

    prompt = (f"请为以下旅行需求提供详细规划信息：\n"
              f"- 目的地：{destination}\n"
              f"- 天数：{days}天\n"
              f"- 风格：{style_map.get(style, style)}\n"
              f"- 兴趣：{interest_str}{season_str}")

    info = _call_llm(
        [{"role": "user", "content": prompt}],
        RESEARCH_SYSTEM,
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

    # 2. 备用：wttr.in
    city_map = {
        '北京': 'Beijing', '上海': 'Shanghai', '广州': 'Guangzhou',
        '深圳': 'Shenzhen', '成都': 'Chengdu', '杭州': 'Hangzhou',
        '西安': "Xi\u0027an", '重庆': 'Chongqing', '南京': 'Nanjing',
        '武汉': 'Wuhan', '长沙': 'Changsha', '昆明': 'Kunming',
        '厦门': 'Xiamen', '青岛': 'Qingdao', '大连': 'Dalian',
        '三亚': 'Sanya', '丽江': 'Lijiang', '桂林': 'Guilin',
        '东京': 'Tokyo', '首尔': 'Seoul', '曼谷': 'Bangkok',
        '新加坡': 'Singapore', '巴黎': 'Paris', '伦敦': 'London',
        '纽约': 'New York', '悉尼': 'Sydney', '迪拜': 'Dubai',
    }

    city_en = city_map.get(destination, destination)

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

    style_map = {
        'budget': '经济实惠型', 'comfortable': '舒适休闲型',
        'luxury': '高端奢华型', 'adventure': '探险刺激型',
        'cultural': '文化深度游', 'beach': '海滨度假型',
    }

    interest_str = '、'.join(interests) if interests else '无特定偏好'
    budget_str = f"\n- 预算：¥{budget}/人" if budget else ""

    prompt = (f"请为以下旅行制定详细行程：\n"
              f"- 目的地：{destination}\n"
              f"- 天数：{days}天\n"
              f"- 风格：{style_map.get(style, style)}\n"
              f"- 兴趣：{interest_str}{budget_str}\n\n"
              f"参考信息：\n{info[:1000]}\n\n"
              f"天气信息：\n{weather}")

    itinerary = _call_llm(
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

    breakdown = _call_llm(
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

    # Combine all sections
    output_parts = []

    if weather:
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

    refined = _call_llm(
        [{"role": "user", "content": f"请优化{destination}的行程"}],
        refine_system,
        max_tokens=1024
    )

    _safe_print(f"[行程优化] 第 {state.get('refinement_round', 1)} 轮 — 已根据反馈调整")
    return {'itinerary': refined, 'refinement_round': state.get('refinement_round', 0) + 1}
