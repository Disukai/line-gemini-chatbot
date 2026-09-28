"""
Conversation memory manager, bot message tracking, and profile cache for LINE chat groups.
Maintains a sliding window of recent group messages, sent bot message IDs, and user display names.
"""
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    sender_id: str
    sender_name: str
    text: str
    is_bot: bool = False
    timestamp: float = field(default_factory=time.time)
    image_desc: Optional[str] = None


class GroupMemory:
    """Maintains recent conversation history and room-specific settings."""
    def __init__(self, max_history: int = 25):
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        self.persona: Optional[str] = None  # Room-specific override if set

    def add_message(
        self,
        sender_id: str,
        sender_name: str,
        text: str,
        is_bot: bool = False,
        image_desc: Optional[str] = None
    ):
        msg = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_bot=is_bot,
            image_desc=image_desc
        )
        self.messages.append(msg)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def clear(self):
        self.messages.clear()

    def get_history_formatted(self, bot_name: str) -> str:
        """Formats the sliding window of messages for Gemini context."""
        lines = []
        for msg in self.messages:
            if msg.is_bot:
                lines.append(f"[{bot_name} (คุณ)]: {msg.text}")
            else:
                extra = f" (ส่งรูป: {msg.image_desc})" if msg.image_desc else ""
                lines.append(f"[{msg.sender_name}]: {msg.text}{extra}")
        return "\n".join(lines)


class MemoryManager:
    """Singleton memory manager for all LINE chats, user profiles, and bot message tracking."""
    def __init__(self, max_history: int = 25):
        self.max_history = max_history
        self.chats: Dict[str, GroupMemory] = {}
        # Cache for user display names: {user_id: (display_name, cached_timestamp, is_fallback)}
        self.profile_cache: Dict[str, Tuple[str, float, bool]] = {}
        self.cache_ttl: float = 3600 * 6  # 6 hours for successfully resolved names
        self.fallback_cache_ttl: float = 30.0  # 30 seconds for temporary fallback names

        # Track sent bot message IDs so quote replies to the bot are recognized
        # while quotes between other group members are not intercepted.
        self._bot_message_ids: Set[str] = set()
        self._bot_message_id_order: List[str] = []
        self._max_tracked_bot_messages: int = 500

    def register_bot_message_id(self, message_id: str):
        """Registers a message ID sent by the bot to detect future quote replies."""
        if not message_id:
            return
        if message_id not in self._bot_message_ids:
            self._bot_message_ids.add(message_id)
            self._bot_message_id_order.append(message_id)
            if len(self._bot_message_id_order) > self._max_tracked_bot_messages:
                oldest = self._bot_message_id_order.pop(0)
                self._bot_message_ids.discard(oldest)

    def is_bot_message(self, message_id: str) -> bool:
        """Checks if a message ID was originated by this bot."""
        if not message_id:
            return False
        return message_id in self._bot_message_ids

    def has_bot_messages(self) -> bool:
        """Returns True if any bot messages have been registered in this session."""
        return len(self._bot_message_ids) > 0

    def get_group_memory(self, chat_id: str) -> GroupMemory:
        if chat_id not in self.chats:
            self.chats[chat_id] = GroupMemory(max_history=self.max_history)
        return self.chats[chat_id]

    def add_message(
        self,
        chat_id: str,
        sender_id: str,
        sender_name: str,
        text: str,
        is_bot: bool = False,
        image_desc: Optional[str] = None
    ):
        mem = self.get_group_memory(chat_id)
        mem.add_message(sender_id, sender_name, text, is_bot, image_desc)

    def clear_memory(self, chat_id: str):
        if chat_id in self.chats:
            self.chats[chat_id].clear()

    def set_persona(self, chat_id: str, persona_key: str):
        mem = self.get_group_memory(chat_id)
        mem.persona = persona_key

    def get_persona(self, chat_id: str, default_persona: str) -> str:
        mem = self.get_group_memory(chat_id)
        return mem.persona or default_persona

    def get_cached_name(self, user_id: str) -> Optional[str]:
        if user_id in self.profile_cache:
            name, cached_time, is_fallback = self.profile_cache[user_id]
            ttl = self.fallback_cache_ttl if is_fallback else self.cache_ttl
            if time.time() - cached_time < ttl:
                return name
        return None

    def cache_name(self, user_id: str, name: str, is_fallback: bool = False):
        self.profile_cache[user_id] = (name, time.time(), is_fallback)


memory_manager = MemoryManager()
