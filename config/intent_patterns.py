"""
Intent classification patterns and response templates.

Centralizes all keyword lists, pattern-matching rules, and canned
responses used for intent classification and quick replies across
the travel planner nodes.
"""

from .destinations import KNOWN_PLACES

# ============================================================
# Quick-match greetings  (skip LLM for trivial hellos / thanks)
# ============================================================
GREETINGS = [
    '你好', '您好', '嗨', 'hi', 'hello', '在吗', '在不在',
    '早上好', '晚上好', '中午好', '下午好', '嘿', 'hey',
    '谢谢', '感谢', '再见', '拜拜', 'bye', '好的', '嗯',
    '哦', '哈哈', '嘻嘻', '嗯嗯', 'ok', 'ok的',
]


# ============================================================
# Pre-canned greeting responses  (fast-path, no LLM needed)
# ============================================================
GREETING_RESPONSES = {
    '你好': [
        '你好呀！👋 想去哪里玩？告诉我目的地和天数，我来帮你规划行程～',
        '嗨！我是你的旅游规划助手 🌍 说说你想去哪？',
    ],
    '您好': [
        '您好！😊 有什么旅行计划吗？我可以帮您做详细的行程规划。',
        '您好呀！想去哪里玩？告诉我目的地和预算就好啦～',
    ],
    'hi': [
        'Hi there! 👋 Ready to plan your next trip? Tell me where you want to go!',
        'Hey! I can help you plan a trip. Where are you thinking of going?',
    ],
    'hello': [
        'Hello! 🌍 Looking for travel inspiration? Just tell me your destination!',
        'Hi! What trip can I help you plan today?',
    ],
    '在吗': [
        '在的！有什么旅行计划需要帮忙的吗？🗺️',
        '在呢～想去哪里玩？告诉我目的地和天数就好！',
    ],
    '谢谢': [
        '不客气！😊 还有其他问题随时问我～',
        '应该的！祝你旅途愉快 🎉',
    ],
    '再见': [
        '再见！旅途愉快～ 🌟',
        '拜拜！下次旅行还找我规划 😄',
    ],
}


# ============================================================
# Recommend-intent keywords
# (user has NO destination yet and wants suggestions)
# ============================================================
RECOMMEND_KEYWORDS = [
    '去哪里', '去哪玩', '推荐', '好去处', '有什么推荐',
    '你觉得', '哪里好', '去哪个', '去什么', '求推荐',
    '想去但不知道', '没有目的地',
]


# ============================================================
# Destination indicators  ("去" + known place)
# If the message contains one of these the user has a destination.
# Derived automatically from KNOWN_PLACES.
# ============================================================
DESTINATION_INDICATORS = ['去' + place for place in KNOWN_PLACES]


# ============================================================
# Interest category → trigger keywords
# ============================================================
INTEREST_KEYWORDS = {
    '美食': ['美食', '吃', '吃货'],
    '海滨': ['海边', '海滩', '海岛', '潜水'],
    '文化': ['历史', '文化', '古迹', '博物馆'],
    '自然': ['自然', '山水', '风景', '户外'],
    '购物': ['购物', '逛街'],
    '亲子': ['亲子', '孩子', '带娃', '家庭'],
}


# ============================================================
# Budget preference keywords
# ============================================================
BUDGET_KEYWORDS = {
    'economic': ['穷游', '省钱', '经济', '便宜'],
    'luxury':   ['奢华', '高端', '豪华', '贵'],
}


# ============================================================
# Season keyword → season name
# ============================================================
SEASON_KEYWORDS = {
    '春天': '春季', '夏天': '夏季', '秋天': '秋季', '冬天': '冬季',
    '寒假': '冬季', '暑假': '夏季', '五一': '春季', '国庆': '秋季',
}


# ============================================================
# Travel style keywords  (style_code → trigger words)
# Used in fallback parsing when LLM JSON extraction fails.
# ============================================================
STYLE_KEYWORDS = {
    'budget':    ['穷游', '省钱', '经济'],
    'luxury':    ['奢华', '高端', '豪华'],
    'adventure': ['探险', '冒险', '户外'],
    'cultural':  ['文化', '历史', '博物馆'],
    'beach':     ['海边', '海滩', '海岛', '度假'],
}


# ============================================================
# Travel style code → Chinese display name
# ============================================================
TRAVEL_STYLE_MAP = {
    'budget':     '经济实惠型',
    'comfortable': '舒适休闲型',
    'luxury':     '高端奢华型',
    'adventure':  '探险刺激型',
    'cultural':   '文化深度游',
    'beach':      '海滨度假型',
}
