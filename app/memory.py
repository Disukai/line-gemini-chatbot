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
    message_id: Optional[str] = None
    quoted_message_id: Optional[str] = None
    image_desc: Optional[str] = None
    file_desc: Optional[str] = None


class GroupMemory:
    """Maintains recent conversation history, indexed messages for quote-replies, and room-specific settings."""
    def __init__(self, max_history: int = 35):
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        self.messages_by_id: Dict[str, ChatMessage] = {}
        self.persona: Optional[str] = None  # Room-specific override if set
        self.trigger_mode: Optional[str] = None  # Room-specific trigger mode override (/mode)
        self.last_bot_reply_time: float = 0.0
        self.messages_since_bot_spoke: int = 0

    def add_message(
        self,
        sender_id: str,
        sender_name: str,
        text: str,
        is_bot: bool = False,
        message_id: Optional[str] = None,
        quoted_message_id: Optional[str] = None,
        image_desc: Optional[str] = None,
        file_desc: Optional[str] = None
    ) -> ChatMessage:
        msg = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_bot=is_bot,
            message_id=message_id,
            quoted_message_id=quoted_message_id,
            image_desc=image_desc,
            file_desc=file_desc
        )
        self.messages.append(msg)
        if message_id:
            self.messages_by_id[message_id] = msg

        if is_bot:
            self.messages_since_bot_spoke = 0
            self.last_bot_reply_time = time.time()
        else:
            self.messages_since_bot_spoke += 1

        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

        # Keep messages_by_id cache bounded
        if len(self.messages_by_id) > 300:
            excess = len(self.messages_by_id) - 300
            for k in list(self.messages_by_id.keys())[:excess]:
                del self.messages_by_id[k]

        return msg

    def get_message_by_id(self, message_id: Optional[str]) -> Optional[ChatMessage]:
        """Retrieves a previously sent message by its LINE message ID."""
        if not message_id:
            return None
        return self.messages_by_id.get(message_id)

    def register_last_bot_message_id(self, message_id: str):
        """Associates the given LINE message ID with the most recently added bot message."""
        if not message_id:
            return
        for msg in reversed(self.messages):
            if msg.is_bot and not msg.message_id:
                msg.message_id = message_id
                self.messages_by_id[message_id] = msg
                break

    def clear(self):
        self.messages.clear()
        self.messages_by_id.clear()

    def get_history_formatted(self, bot_name: str) -> str:
        """Formats the sliding window of messages for Gemini context, showing quote replies clearly."""
        lines = []
        for msg in self.messages:
            quoted_info = ""
            if msg.quoted_message_id:
                q_msg = self.get_message_by_id(msg.quoted_message_id)
                if q_msg:
                    snippet = q_msg.text[:40].replace("\n", " ") + ("..." if len(q_msg.text) > 40 else "")
                    q_author = f"{bot_name} (คุณ)" if q_msg.is_bot else q_msg.sender_name
                    quoted_info = f" (ตอบกลับ {q_author}: \"{snippet}\")"
                else:
                    quoted_info = " (ตอบกลับข้อความเดิม)"

            extras = []
            if msg.image_desc:
                extras.append(f"รูปภาพ: {msg.image_desc}")
            if msg.file_desc:
                extras.append(f"ไฟล์แนบ: {msg.file_desc}")
            extra_str = f" [{', '.join(extras)}]" if extras else ""

            if msg.is_bot:
                lines.append(f"[{bot_name} (คุณ){quoted_info}]: {msg.text}{extra_str}")
            else:
                lines.append(f"[{msg.sender_name}{quoted_info}]: {msg.text}{extra_str}")
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

    def register_bot_message_id(self, message_id: str, chat_id: Optional[str] = None):
        """Registers a message ID sent by the bot to detect future quote replies and binds to last bot message."""
        if not message_id:
            return
        if message_id not in self._bot_message_ids:
            self._bot_message_ids.add(message_id)
            self._bot_message_id_order.append(message_id)
            if len(self._bot_message_id_order) > self._max_tracked_bot_messages:
                oldest = self._bot_message_id_order.pop(0)
                self._bot_message_ids.discard(oldest)

        if chat_id and chat_id in self.chats:
            self.chats[chat_id].register_last_bot_message_id(message_id)

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
        message_id: Optional[str] = None,
        quoted_message_id: Optional[str] = None,
        image_desc: Optional[str] = None,
        file_desc: Optional[str] = None
    ) -> ChatMessage:
        mem = self.get_group_memory(chat_id)
        return mem.add_message(
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_bot=is_bot,
            message_id=message_id,
            quoted_message_id=quoted_message_id,
            image_desc=image_desc,
            file_desc=file_desc
        )

    def get_message(self, chat_id: str, message_id: Optional[str]) -> Optional[ChatMessage]:
        """Looks up a specific message by its message ID in the given chat."""
        if not chat_id or not message_id:
            return None
        mem = self.get_group_memory(chat_id)
        return mem.get_message_by_id(message_id)

    def clear_memory(self, chat_id: str):
        if chat_id in self.chats:
            self.chats[chat_id].clear()

    def set_persona(self, chat_id: str, persona_key: str):
        mem = self.get_group_memory(chat_id)
        mem.persona = persona_key

    def get_persona(self, chat_id: str, default_persona: str) -> str:
        mem = self.get_group_memory(chat_id)
        return mem.persona or default_persona

    def set_trigger_mode(self, chat_id: str, mode: str):
        mem = self.get_group_memory(chat_id)
        mem.trigger_mode = mode

    def get_trigger_mode(self, chat_id: str, default_mode: str) -> str:
        mem = self.get_group_memory(chat_id)
        return mem.trigger_mode or default_mode

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
