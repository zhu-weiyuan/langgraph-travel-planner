"""
User Preference Memory Module

Stores and retrieves user travel preferences across sessions:
- Budget range
- Travel style (adventure, relaxation, cultural, etc.)
- Accommodation preference
- Dietary restrictions
- Previous destinations

Uses simple JSON file storage for persistence.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# Storage path
MEMORY_FILE = Path(__file__).parent.parent / "user_memory.json"


class UserMemory:
    """Simple user preference memory with JSON persistence."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.preferences = self._load()

    def _load(self) -> Dict:
        """Load preferences from file."""
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                return data.get(self.user_id, {})
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        """Save preferences to file."""
        if not MEMORY_FILE.exists():
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load all users' data
        all_data = {}
        if MEMORY_FILE.exists():
            try:
                all_data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass

        all_data[self.user_id] = self.preferences
        MEMORY_FILE.write_text(
            json.dumps(all_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def set_preference(self, key: str, value):
        """Set a user preference."""
        self.preferences[key] = value
        self._save()

    def get_preference(self, key: str, default=None):
        """Get a user preference."""
        return self.preferences.get(key, default)

    def get_all(self) -> Dict:
        """Get all preferences."""
        return dict(self.preferences)

    def add_destination(self, destination: str, rating: Optional[int] = None,
                       notes: Optional[str] = None):
        """Record a visited destination."""
        if "destinations" not in self.preferences:
            self.preferences["destinations"] = []

        entry = {
            "destination": destination,
            "rating": rating,
            "notes": notes,
            "visited_at": datetime.now().isoformat()
        }
        self.preferences["destinations"].append(entry)
        self._save()

    def get_destinations(self) -> List[Dict]:
        """Get all visited destinations."""
        return self.preferences.get("destinations", [])

    def format_context(self) -> str:
        """Format preferences as context for LLM prompt."""
        if not self.preferences:
            return ""

        lines = ["\n## 用户偏好（历史记录）"]
        pref_map = {
            "budget": "预算范围",
            "style": "旅行风格",
            "accommodation": "住宿偏好",
            "dietary": "饮食限制",
            "companions": "同行人员",
        }

        for key, label in pref_map.items():
            value = self.preferences.get(key)
            if value:
                lines.append(f"- {label}: {value}")

        destinations = self.preferences.get("destinations", [])
        if destinations:
            lines.append(f"\n- 去过{len(destinations)}个目的地")
            for d in destinations[-3:]:  # Last 3
                rating = f" ({d['rating']}/5)" if d.get("rating") else ""
                lines.append(f"  - {d['destination']}{rating}")

        return "\n".join(lines)


# Global instance
_memory = None


def get_memory(user_id: str = "default") -> UserMemory:
    """Get or create user memory instance."""
    global _memory
    if _memory is None or _memory.user_id != user_id:
        _memory = UserMemory(user_id)
    return _memory
