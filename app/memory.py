"""
Conversation memory manager, bot message tracking, and profile cache for LINE chat groups.
Maintains unified persistent memory across all 1-on-1 private chats and group chats,
ensuring facts, messages, and quote-replies are synced across environments.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("line_gemini_bot")
UNIFIED_MEMORY_FILE = Path("data/unified_memory.json")
BOT_MSG_IDS_FILE = Path("data/bot_message_ids.json")


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
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None  # "group" or "user"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        known_keys = {
            "sender_id", "sender_name", "text", "is_bot",
            "timestamp", "message_id", "quoted_message_id",
            "image_desc", "file_desc", "chat_id", "chat_type"
        }
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


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
        file_desc: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> ChatMessage:
        msg = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_bot=is_bot,
            timestamp=timestamp if timestamp is not None else time.time(),
            message_id=message_id,
            quoted_message_id=quoted_message_id,
            image_desc=image_desc,
            file_desc=file_desc,
            chat_id=chat_id,
            chat_type=chat_type
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
    """Singleton memory manager providing unified, persistent cross-chat memory between private chats and groups."""
    def __init__(self, max_history: int = 35):
        self.max_history = max_history
        self.chats: Dict[str, GroupMemory] = {}
        # Cache for user display names: {user_id: (display_name, cached_timestamp, is_fallback)}
        self.profile_cache: Dict[str, Tuple[str, float, bool]] = {}
        self.cache_ttl: float = 3600 * 6  # 6 hours for successfully resolved names
        self.fallback_cache_ttl: float = 30.0  # 30 seconds for temporary fallback names

        # Global cross-chat index across all rooms and 1-on-1 chats
        self.global_messages_by_id: Dict[str, ChatMessage] = {}
        self.global_recent_messages: List[ChatMessage] = []
        self.max_global_messages: int = 150

        # Shared user knowledge (facts learned about users across all environments)
        # {user_id: {"name": str, "facts": List[str], "recent_topics": List[str]}}
        self.user_knowledge: Dict[str, Dict] = {}

        # Track sent bot message IDs so quote replies to the bot are recognized
        self._bot_message_ids: Set[str] = set()
        self._bot_message_id_order: List[str] = []
        self._max_tracked_bot_messages: int = 1000

        self._load_all_memory()

    def _load_all_memory(self):
        """Loads unified memory and bot message IDs from disk to survive container restarts."""
        # 1. Load bot message IDs
        try:
            if BOT_MSG_IDS_FILE.exists():
                with open(BOT_MSG_IDS_FILE, "r", encoding="utf-8") as f:
                    ids = json.load(f)
                    if isinstance(ids, list):
                        self._bot_message_id_order = ids[-self._max_tracked_bot_messages:]
                        self._bot_message_ids = set(self._bot_message_id_order)
        except Exception as e:
            logger.warning("Failed to load bot message IDs: %s", e)

        # 2. Load unified persistent memory
        try:
            if UNIFIED_MEMORY_FILE.exists():
                with open(UNIFIED_MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load global messages
                global_msgs = data.get("global_messages", [])
                for item in global_msgs:
                    msg = ChatMessage.from_dict(item)
                    if msg.message_id:
                        self.global_messages_by_id[msg.message_id] = msg
                    self.global_recent_messages.append(msg)
                    if msg.is_bot and msg.message_id:
                        self._bot_message_ids.add(msg.message_id)

                self.global_recent_messages = self.global_recent_messages[-self.max_global_messages:]

                # Load chats
                saved_chats = data.get("chats", {})
                for cid, c_data in saved_chats.items():
                    mem = self.get_group_memory(cid)
                    mem.persona = c_data.get("persona")
                    mem.trigger_mode = c_data.get("trigger_mode")
                    for m_dict in c_data.get("messages", []):
                        m_obj = ChatMessage.from_dict(m_dict)
                        mem.messages.append(m_obj)
                        if m_obj.message_id:
                            mem.messages_by_id[m_obj.message_id] = m_obj
                    mem.messages = mem.messages[-self.max_history:]

                # Load user knowledge
                self.user_knowledge = data.get("user_knowledge", {})
                logger.info(
                    "Unified memory loaded successfully: %d chats, %d global messages, %d user profiles",
                    len(self.chats), len(self.global_messages_by_id), len(self.user_knowledge)
                )
        except Exception as e:
            logger.warning("Failed to load unified memory: %s", e)

    def _save_all_memory(self):
        """Persists unified memory and bot message IDs to disk."""
        try:
            UNIFIED_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            serialized_chats = {}
            for cid, mem in self.chats.items():
                serialized_chats[cid] = {
                    "persona": mem.persona,
                    "trigger_mode": mem.trigger_mode,
                    "messages": [m.to_dict() for m in mem.messages[-self.max_history:]]
                }

            recent_global = [m.to_dict() for m in self.global_recent_messages[-self.max_global_messages:]]

            payload = {
                "chats": serialized_chats,
                "global_messages": recent_global,
                "user_knowledge": self.user_knowledge,
                "updated_at": time.time()
            }
            tmp_unified = UNIFIED_MEMORY_FILE.with_suffix(".tmp")
            with open(tmp_unified, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp_unified.replace(UNIFIED_MEMORY_FILE)

            tmp_ids = BOT_MSG_IDS_FILE.with_suffix(".tmp")
            with open(tmp_ids, "w", encoding="utf-8") as f:
                json.dump(self._bot_message_id_order, f)
            tmp_ids.replace(BOT_MSG_IDS_FILE)
        except Exception as e:
            logger.warning("Failed to save unified memory: %s", e)

    def register_bot_message_id(self, message_id: str, chat_id: Optional[str] = None):
        """Registers a message ID sent by the bot to detect future quote replies."""
        if not message_id:
            return
        if message_id not in self._bot_message_ids:
            self._bot_message_ids.add(message_id)
            self._bot_message_id_order.append(message_id)
            if len(self._bot_message_id_order) > self._max_tracked_bot_messages:
                oldest = self._bot_message_id_order.pop(0)
                self._bot_message_ids.discard(oldest)
            self._save_all_memory()

        if chat_id and chat_id in self.chats:
            self.chats[chat_id].register_last_bot_message_id(message_id)

    def is_bot_message(self, message_id: str) -> bool:
        """Checks if a message ID was originated by this bot."""
        if not message_id:
            return False
        return message_id in self._bot_message_ids

    def has_bot_messages(self) -> bool:
        """Returns True if any bot messages have been registered or loaded."""
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
        file_desc: Optional[str] = None,
        chat_type: Optional[str] = None
    ) -> ChatMessage:
        mem = self.get_group_memory(chat_id)
        msg = mem.add_message(
            sender_id=sender_id,
            sender_name=sender_name,
            text=text,
            is_bot=is_bot,
            message_id=message_id,
            quoted_message_id=quoted_message_id,
            image_desc=image_desc,
            file_desc=file_desc,
            chat_id=chat_id,
            chat_type=chat_type
        )

        if message_id:
            self.global_messages_by_id[message_id] = msg
            if len(self.global_messages_by_id) > 1000:
                del_key = next(iter(self.global_messages_by_id))
                del self.global_messages_by_id[del_key]

        self.global_recent_messages.append(msg)
        if len(self.global_recent_messages) > self.max_global_messages:
            self.global_recent_messages = self.global_recent_messages[-self.max_global_messages:]

        if not is_bot and sender_id and sender_id != "unknown":
            if sender_id not in self.user_knowledge:
                self.user_knowledge[sender_id] = {
                    "name": sender_name,
                    "facts": [],
                    "recent_topics": []
                }
            elif sender_name and sender_name != "เพื่อน":
                self.user_knowledge[sender_id]["name"] = sender_name

            self._extract_facts_from_text(sender_id, sender_name, text)

        self._save_all_memory()
        return msg

    def _extract_facts_from_text(self, user_id: str, sender_name: str, text: str):
        """Extracts notable personal statements or plans to remember long term."""
        clean = text.strip()
        triggers = [
            "เราชอบ", "ผมชอบ", "กูชอบ", "เราทำงาน", "ผมทำงาน", "กูทำงาน",
            "เราอยู่", "ผมอยู่", "กูอยู่", "เรามี", "ผมมี", "กูมี",
            "เราเลี้ยง", "ผมเลี้ยง", "กูเลี้ยง", "เราชื่อ", "ผมชื่อ",
            "จำไว้ด้วยว่า", "จำไว้นะ", "บอกโทมัสไว้ก่อน", "อย่าลืมนะ"
        ]
        for tr in triggers:
            if tr in clean:
                fact_snippet = clean[:80]
                facts_list = self.user_knowledge[user_id].setdefault("facts", [])
                if fact_snippet not in facts_list:
                    facts_list.append(fact_snippet)
                    if len(facts_list) > 10:
                        facts_list.pop(0)
                break

    def get_message(self, chat_id: Optional[str], message_id: Optional[str]) -> Optional[ChatMessage]:
        """Looks up a specific message by its message ID in the given chat or globally across all chats."""
        if not message_id:
            return None
        if chat_id and chat_id in self.chats:
            msg = self.chats[chat_id].get_message_by_id(message_id)
            if msg:
                return msg

        if message_id in self.global_messages_by_id:
            return self.global_messages_by_id[message_id]

        return None

    def get_cross_chat_context(
        self,
        current_chat_id: str,
        sender_id: Optional[str] = None,
        max_messages: int = 10
    ) -> str:
        """
        Builds synchronized cross-chat context between 1-on-1 private chat and groups.
        Allows Thomas to remember things discussed across all environments.
        """
        lines = []

        if sender_id and sender_id in self.user_knowledge:
            uk = self.user_knowledge[sender_id]
            u_name = uk.get("name", "เพื่อน")
            facts = uk.get("facts", [])
            if facts:
                lines.append(f"- ข้อมูลสำคัญเกี่ยวกับ {u_name} (จำได้จากทุกแชท): {', '.join(facts[-5:])}")

        other_msgs = [
            m for m in self.global_recent_messages
            if m.chat_id and m.chat_id != current_chat_id
        ]

        if other_msgs:
            recent_others = other_msgs[-max_messages:]
            lines.append("- ข้อความล่าสุดจากแชทอื่นๆ (แชทส่วนตัว / กลุ่มอื่นที่เชื่อมข้อมูลกัน):")
            for m in recent_others:
                location = "แชทส่วนตัว" if m.chat_type == "user" else "กลุ่มไลน์"
                speaker = f"{m.sender_name} (ใน{location})" if not m.is_bot else f"Thomas (คุณ ใน{location})"
                clean_t = m.text.replace("\n", " ")[:60]
                lines.append(f"  • [{speaker}]: \"{clean_t}\"")

        if not lines:
            return ""

        header = "[ความจำเชื่อมโยงข้ามแชท (ข้อมูลทุกอย่างในแชทส่วนตัวและกลุ่มไลน์ซิงค์ถึงกัน)]:"
        return header + "\n" + "\n".join(lines)

    def clear_memory(self, chat_id: str):
        if chat_id in self.chats:
            self.chats[chat_id].clear()
            self._save_all_memory()

    def set_persona(self, chat_id: str, persona_key: str):
        mem = self.get_group_memory(chat_id)
        mem.persona = persona_key
        self._save_all_memory()

    def get_persona(self, chat_id: str, default_persona: str) -> str:
        mem = self.get_group_memory(chat_id)
        return mem.persona or default_persona

    def set_trigger_mode(self, chat_id: str, mode: str):
        mem = self.get_group_memory(chat_id)
        mem.trigger_mode = mode
        self._save_all_memory()

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
