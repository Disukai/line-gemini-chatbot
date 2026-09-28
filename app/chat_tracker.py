"""
Persistent storage for active LINE chats and rooms.
Allows background schedulers (morning news, spontaneous conversation starters)
to know which groups to broadcast to, even after server restarts or deployments.
"""
import os
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("line_gemini_bot")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STORAGE_FILE = os.path.join(DATA_DIR, "active_chats.json")


class ChatTracker:
    def __init__(self):
        self._chats: Dict[str, dict] = {}
        self._load()

    def _ensure_data_dir(self):
        if not os.path.exists(DATA_DIR):
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create data directory: %s", e)

    def _load(self):
        self._ensure_data_dir()
        if os.path.exists(STORAGE_FILE):
            try:
                with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                    self._chats = json.load(f)
                logger.info("Loaded %d active chats from %s", len(self._chats), STORAGE_FILE)
            except Exception as e:
                logger.error("Failed to load active chats from %s: %s", STORAGE_FILE, e)
                self._chats = {}
        else:
            self._chats = {}

    def _save(self):
        self._ensure_data_dir()
        try:
            with open(STORAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._chats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save active chats to %s: %s", STORAGE_FILE, e)

    def register_chat(self, chat_id: str, chat_type: str = "group", name: Optional[str] = None):
        """Registers or updates a chat as active."""
        if not chat_id:
            return

        is_new = chat_id not in self._chats
        info = self._chats.get(chat_id, {
            "type": chat_type,
            "name": name or "",
            "schedule_enabled": True,
            "created_at": None
        })

        if name and not info.get("name"):
            info["name"] = name
        info["type"] = chat_type

        self._chats[chat_id] = info
        if is_new:
            logger.info("Registered new active %s [%s]", chat_type, chat_id)
            self._save()

    def unregister_chat(self, chat_id: str):
        """Removes a chat (e.g. when bot leaves group)."""
        if chat_id in self._chats:
            del self._chats[chat_id]
            self._save()
            logger.info("Unregistered chat [%s]", chat_id)

    def set_schedule_enabled(self, chat_id: str, enabled: bool):
        if chat_id in self._chats:
            self._chats[chat_id]["schedule_enabled"] = enabled
            self._save()

    def is_schedule_enabled(self, chat_id: str) -> bool:
        if chat_id in self._chats:
            return self._chats[chat_id].get("schedule_enabled", True)
        return True

    def get_active_group_ids(self) -> List[str]:
        """Returns all registered group/room chat IDs that have scheduling enabled."""
        result = []
        for cid, info in self._chats.items():
            # Broadcast to groups/rooms where schedule_enabled is True
            if info.get("type") in ("group", "room") and info.get("schedule_enabled", True):
                result.append(cid)
        return result

    def get_all_registered_chat_ids(self) -> List[str]:
        return list(self._chats.keys())


chat_tracker = ChatTracker()
