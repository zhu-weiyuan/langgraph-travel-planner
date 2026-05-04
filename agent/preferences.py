# -*- coding: utf-8 -*-
"""
User travel preference memory.

Stores and retrieves user preferences across sessions:
- cuisine preferences (spicy, light, local)
- budget range (economy, mid-range, luxury)
- travel style (nature, city, culture, adventure)
- accommodation preference
- visited cities (avoid recommending again)

Persistence: SQLite (preferences.db)
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

_DB_PATH = Path(__file__).parent.parent / "preferences.db"

# Keywords for inferring preferences from user messages
_CUISINE_KEYWORDS = {
    "spicy": ["辣", "麻辣", "川菜", "火锅", "湘菜"],
    "light": ["清淡", "粤菜", "点心", "早茶"],
    "local": ["地道", "本地", "特色", "小吃"],
}

_BUDGET_KEYWORDS = {
    "economy": ["便宜", "经济", "穷游", "省钱", "预算有限"],
    "mid-range": ["适中", "中等", "性价比"],
    "luxury": ["豪华", "高端", "五星", "不差钱", "奢侈"],
}

_STYLE_KEYWORDS = {
    "nature": ["自然", "山水", "户外", "徒步", "森林"],
    "city": ["城市", "购物", "美食", "夜景"],
    "culture": ["文化", "历史", "博物馆", "古迹", "人文"],
    "adventure": ["冒险", "刺激", "极限", "探险"],
}


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection):
    """Initialize the preferences database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            session_id TEXT NOT NULL,
            pref_key TEXT NOT NULL,
            pref_value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, pref_key)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visited_cities (
            session_id TEXT NOT NULL,
            city TEXT NOT NULL,
            visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (session_id, city)
        )
    """)
    conn.commit()


def save_preference(session_id: str, key: str, value: str):
    """Save or update a user preference.

    Args:
        session_id: User session identifier
        key: Preference key (cuisine, budget, style, accommodation)
        value: Preference value
    """
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO preferences (session_id, pref_key, pref_value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (session_id, key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_preferences(session_id: str) -> Dict[str, str]:
    """Get all preferences for a session.

    Args:
        session_id: User session identifier

    Returns:
        Dict of {pref_key: pref_value}
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT pref_key, pref_value FROM preferences WHERE session_id = ?",
            (session_id,),
        )
        return dict(cursor.fetchall())
    finally:
        conn.close()


def add_visited_city(session_id: str, city: str):
    """Record a visited city to avoid recommending again.

    Args:
        session_id: User session identifier
        city: City name
    """
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO visited_cities (session_id, city) VALUES (?, ?)",
            (session_id, city),
        )
        conn.commit()
    finally:
        conn.close()


def get_visited_cities(session_id: str) -> List[str]:
    """Get list of visited cities for a session.

    Args:
        session_id: User session identifier

    Returns:
        List of city names.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT city FROM visited_cities WHERE session_id = ? ORDER BY visited_at DESC",
            (session_id,),
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def infer_preference(message: str) -> Dict[str, str]:
    """Infer user preferences from a message.

    Args:
        message: User's input message

    Returns:
        Dict of inferred preferences {key: value}
    """
    prefs = {}
    msg_lower = message.lower()

    # Cuisine preference
    for cuisine, keywords in _CUISINE_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            prefs["cuisine"] = cuisine
            break

    # Budget preference
    for budget, keywords in _BUDGET_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            prefs["budget"] = budget
            break

    # Travel style
    for style, keywords in _STYLE_KEYWORDS.items():
        if any(kw in msg_lower for kw in keywords):
            prefs["style"] = style
            break

    return prefs


def build_preference_context(session_id: str) -> str:
    """Build preference context string for system prompt injection.

    Args:
        session_id: User session identifier

    Returns:
        Formatted preference context string, or empty string if no preferences.
    """
    prefs = get_preferences(session_id)
    visited = get_visited_cities(session_id)

    if not prefs and not visited:
        return ""

    lines = ["\n## 用户偏好（历史记忆）\n"]

    if prefs:
        cuisine_names = {"spicy": "辛辣", "light": "清淡", "local": "地道小吃"}
        budget_names = {"economy": "经济型", "mid-range": "中等消费", "luxury": "高端奢华"}
        style_names = {"nature": "自然风光", "city": "城市探索", "culture": "历史文化", "adventure": "冒险刺激"}

        if "cuisine" in prefs:
            lines.append(f"- 饮食偏好：{cuisine_names.get(prefs['cuisine'], prefs['cuisine'])}")
        if "budget" in prefs:
            lines.append(f"- 消费水平：{budget_names.get(prefs['budget'], prefs['budget'])}")
        if "style" in prefs:
            lines.append(f"- 旅行风格：{style_names.get(prefs['style'], prefs['style'])}")
        if "accommodation" in prefs:
            lines.append(f"- 住宿偏好：{prefs['accommodation']}")

    if visited:
        lines.append(f"- 已去过的城市：{', '.join(visited[:5])}")
        lines.append("- 请避免推荐已去过的城市，或说明与之前不同的玩法")

    return "\n".join(lines)
