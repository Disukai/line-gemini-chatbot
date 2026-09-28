"""
Core event processor and LINE Messaging API dispatcher.
Handles text messages, image understanding, group trigger rules, and natural replies.
"""
import asyncio
import logging
import random
import time
from typing import Optional, Tuple, List

from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    ShowLoadingAnimationRequest,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    JoinEvent,
    FollowEvent,
)

from app.config import settings
from app.memory import memory_manager
from app.persona import build_system_prompt
from app.commands import parse_command, execute_command
from app.gemini_client import gemini_client

logger = logging.getLogger("line_gemini_bot")

# Initialize LINE Messaging API configuration
line_config = Configuration(access_token=settings.line_channel_access_token)

# Cached bot user ID to verify mentions
_cached_bot_user_id: Optional[str] = None


class TriggerResult(tuple):
    """
    Result of should_trigger_response: (should_reply: bool, is_spontaneous: bool).
    Supports tuple unpacking:
        should_reply, is_spontaneous = should_trigger_response(...)
    and named property access:
        result.should_reply, result.is_spontaneous
    """
    def __new__(cls, should_reply: bool, is_spontaneous: bool = False):
        return super().__new__(cls, (bool(should_reply), bool(is_spontaneous)))

    @property
    def should_reply(self) -> bool:
        return self[0]

    @property
    def is_spontaneous(self) -> bool:
        return self[1]

    def __bool__(self) -> bool:
        return self[0]


def get_api_client() -> ApiClient:
    return ApiClient(line_config)


def get_bot_user_id(api_client: ApiClient) -> Optional[str]:
    """Retrieves and caches the bot's own LINE user ID via get_bot_info."""
    global _cached_bot_user_id
    if _cached_bot_user_id:
        return _cached_bot_user_id
    if not settings.line_channel_access_token:
        return None
    try:
        messaging_api = MessagingApi(api_client)
        info = messaging_api.get_bot_info()
        if info and getattr(info, "user_id", None):
            _cached_bot_user_id = info.user_id
            logger.info("LINE Bot user ID resolved: %s", _cached_bot_user_id)
            return _cached_bot_user_id
    except Exception as e:
        logger.debug("Could not resolve bot user ID: %s", e)
    return None


def extract_chat_and_user_ids(event) -> Tuple[str, Optional[str], bool]:
    """
    Extracts (chat_id, user_id, is_group) from a LINE webhook event.
    """
    source = getattr(event, "source", None)
    if not source:
        return "", None, False

    source_type = getattr(source, "type", "user")

    if source_type == "group":
        return getattr(source, "group_id", ""), getattr(source, "user_id", None), True
    elif source_type == "room":
        return getattr(source, "room_id", ""), getattr(source, "user_id", None), True
    else:
        user_id = getattr(source, "user_id", "")
        return user_id, user_id, False


def resolve_sender_name(api_client: ApiClient, chat_id: str, user_id: Optional[str], is_group: bool) -> str:
    """
    Retrieves the user's display name, caching results to avoid LINE rate limits.
    """
    if not user_id:
        return "เพื่อน"

    cached = memory_manager.get_cached_name(user_id)
    if cached:
        return cached

    messaging_api = MessagingApi(api_client)

    # 1. Try group member profile if in group
    if is_group and chat_id:
        try:
            profile = messaging_api.get_group_member_profile(chat_id, user_id)
            if profile and profile.display_name:
                memory_manager.cache_name(user_id, profile.display_name, is_fallback=False)
                return profile.display_name
        except Exception as e:
            logger.debug("Could not get group member profile: %s", e)

    # 2. Try generic user profile
    try:
        profile = messaging_api.get_profile(user_id)
        if profile and profile.display_name:
            memory_manager.cache_name(user_id, profile.display_name, is_fallback=False)
            return profile.display_name
    except Exception as e:
        logger.debug("Could not get user profile: %s", e)

    # If profile retrieval failed, temporarily cache default fallback for 30 seconds
    default_name = "เพื่อน"
    memory_manager.cache_name(user_id, default_name, is_fallback=True)
    return default_name


def chunk_line_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Splits long response text into chunks of <= max_chunk_size,
    respecting LINE's 5,000 character limit per TextMessage and returning at most 5 messages.
    """
    if not text:
        return [""]
    if len(text) <= max_chunk_size:
        return [text]

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += ("\n" if current_chunk else "") + paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(paragraph) > max_chunk_size:
                for i in range(0, len(paragraph), max_chunk_size):
                    chunks.append(paragraph[i:i + max_chunk_size])
                current_chunk = ""
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    # LINE limits reply_message to at most 5 messages
    return chunks[:5]


def should_trigger_response(
    text: str,
    event: MessageEvent,
    is_group: bool,
    bot_name: str,
    nicknames: list[str],
    trigger_mode: str,
    bot_user_id: Optional[str] = None,
    chat_id: Optional[str] = None
) -> TriggerResult:
    """
    Determines if the bot should speak up in the conversation.
    Returns TriggerResult(should_reply, is_spontaneous).
    Direct calls (mentions, nicknames, quote replies, commands) always trigger with is_spontaneous=False.
    Spontaneous chime-ins in groups trigger with is_spontaneous=True based on probability & cooldowns.
    """
    # 1. Always respond in 1-on-1 private chat
    if not is_group:
        return TriggerResult(True, False)

    clean_text = text.strip()

    # 2. Always respond to slash commands (/plan, /boost, /goal, /mode, etc.)
    if clean_text.startswith("/"):
        return TriggerResult(True, False)

    # 3. If mode is "all", respond to everything (not recommended for busy groups)
    if trigger_mode == "all":
        return TriggerResult(True, False)

    # 4. Check if user quoted/replied to a message
    if hasattr(event.message, "quoted_message_id") and event.message.quoted_message_id:
        quoted_id = event.message.quoted_message_id
        # In a group chat, verify that the quoted message was sent by this bot!
        # This prevents the bot from intruding when two group members quote-reply to each other.
        if memory_manager.has_bot_messages():
            if memory_manager.is_bot_message(quoted_id):
                return TriggerResult(True, False)
        else:
            # If no bot messages have been registered yet (e.g. test or startup),
            # check if the text addresses the bot or bot name is present
            lower_text = clean_text.lower()
            all_names = [bot_name.lower()] + [n.lower() for n in nicknames]
            if any(n in lower_text for n in all_names):
                return TriggerResult(True, False)

    # 5. Check if BOT was mentioned via LINE mentionees (do NOT trigger if Alice mentions Bob!)
    if hasattr(event.message, "mention") and event.message.mention:
        mentionees = getattr(event.message.mention, "mentionees", [])
        for m in mentionees:
            m_type = getattr(m, "type", "")
            is_self = getattr(m, "is_self", False)
            uid = getattr(m, "user_id", None)
            if is_self:
                return TriggerResult(True, False)
            if bot_user_id and uid == bot_user_id:
                return TriggerResult(True, False)
            if m_type == "all":
                return TriggerResult(True, False)

    # 6. Check if text contains bot's name or any nickname
    lower_text = clean_text.lower()
    all_names = [bot_name.lower()] + [n.lower() for n in nicknames]
    for name in all_names:
        if name and name in lower_text:
            return TriggerResult(True, False)

    # 7. Smart trigger mode: check for direct questions or requests for help
    if trigger_mode == "smart":
        smart_keywords = [
            "มั้ย", "ไหม", "อะไร", "ใคร", "ที่ไหน", "ยังไง", "ทำไม",
            "รึเปล่า", "ปะ", "ป่ะ", "ช่วยด้วย", "ช่วยคิด", "มีความเห็นว่า",
            "แนะนำหน่อย", "ขอไอเดีย", "ใครรู้บ้าง"
        ]
        if any(kw in lower_text for kw in smart_keywords) or lower_text.endswith("?"):
            return TriggerResult(True, True)

    # 8. Natural Spontaneous Chime-in mode ("chime_in" or "auto")
    if trigger_mode in ("chime_in", "auto"):
        # Trivial message filter: ignore extremely short or acknowledgment-only texts
        if len(clean_text) < 2 or clean_text.lower() in ("ok", "k", "เค", "คับ", "ครับ", "ค่ะ", ".", "!", "?", "55"):
            return TriggerResult(False, False)

        # Anti-spam guard: Check memory cooldown & minimum message gap
        if chat_id:
            mem = memory_manager.get_group_memory(chat_id)
            now = time.time()
            if mem.messages_since_bot_spoke < settings.spontaneous_min_messages:
                return TriggerResult(False, False)
            if mem.last_bot_reply_time > 0 and (now - mem.last_bot_reply_time) < settings.spontaneous_cooldown_seconds:
                return TriggerResult(False, False)

        # Calculate dynamic chime-in probability
        prob = settings.spontaneous_base_rate

        question_keywords = [
            "มั้ย", "ไหม", "อะไร", "ใคร", "ที่ไหน", "ยังไง", "ทำไม",
            "รึเปล่า", "ปะ", "ป่ะ", "ช่วยคิด", "แนะนำหน่อย", "ขอไอเดีย",
            "ใครรู้บ้าง", "ดีไหม", "ดีมั้ย", "กินไร", "ทำไร", "ไปไหน", "เอาไง"
        ]
        if any(kw in lower_text for kw in question_keywords) or clean_text.endswith("?"):
            prob += settings.spontaneous_question_bonus

        slang_keywords = [
            "555", "เหี้ย", "สัส", "โคตร", "วะ", "เว้ย", "งง", "สด๊าว",
            "ชิบหาย", "กู", "มึง", "สุดจัด", "บ้า", "ตาย", "ฟิน", "จริงดิ"
        ]
        if any(kw in lower_text for kw in slang_keywords):
            prob += settings.spontaneous_slang_bonus

        prob = min(prob, 0.70)
        roll = random.random()
        if roll < prob:
            logger.info(
                "Spontaneous chime-in triggered for [%s] in chat [%s] (roll=%.2f < prob=%.2f)",
                bot_name, chat_id, roll, prob
            )
            return TriggerResult(True, True)
        else:
            logger.debug(
                "Spontaneous chime-in passed in chat [%s] (roll=%.2f >= prob=%.2f)",
                chat_id, roll, prob
            )

    return TriggerResult(False, False)


async def process_text_message(event: MessageEvent):
    """Handles an incoming text message event asynchronously and non-blockingly."""
    text = event.message.text
    chat_id, user_id, is_group = extract_chat_and_user_ids(event)

    with get_api_client() as api_client:
        sender_name = await asyncio.to_thread(resolve_sender_name, api_client, chat_id, user_id, is_group)
        current_trigger_mode = memory_manager.get_trigger_mode(chat_id, settings.group_trigger_mode)

        decision = should_trigger_response(
            text=text,
            event=event,
            is_group=is_group,
            bot_name=settings.bot_name,
            nicknames=settings.bot_nicknames,
            trigger_mode=current_trigger_mode,
            bot_user_id=bot_user_id,
            chat_id=chat_id
        )
        should_reply, is_spontaneous = decision.should_reply, decision.is_spontaneous

        # Record incoming message in memory buffer
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id=user_id or "unknown",
            sender_name=sender_name,
            text=text,
            is_bot=False
        )

        if not should_reply:
            logger.info("Message saved to history buffer without reply: [%s]: %s", sender_name, text)
            return

        messaging_api = MessagingApi(api_client)

        # Trigger loading animation in LINE in background (supported in 1-on-1 chats only)
        if settings.enable_loading_animation and not is_group and user_id:
            def _trigger_animation():
                try:
                    anim_req = ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=15)
                    messaging_api.show_loading_animation(anim_req)
                except Exception as e:
                    logger.debug("Loading animation request ignored: %s", e)
            asyncio.create_task(asyncio.to_thread(_trigger_animation))

        # Check for slash commands
        parsed_cmd = parse_command(text)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)

        if parsed_cmd:
            cmd, args = parsed_cmd
            reply_text = await asyncio.to_thread(
                execute_command,
                cmd=cmd,
                args=args,
                chat_id=chat_id,
                sender_name=sender_name,
                history_context=history_context
            )
        else:
            persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
            system_instruction = build_system_prompt(settings.bot_name, persona_key)

            if is_spontaneous:
                user_msg = (
                    f"{text}\n\n"
                    f"[หมายเหตุสำหรับ {settings.bot_name}: คุณไม่ได้ถูกแท็กโดยตรง แต่คุณได้ยินเพื่อนคุยกันในกลุ่ม "
                    f"ให้ตอบแจมหรือแซวสั้นๆ คมๆ 1-2 ประโยค แบบเพื่อนสนิทที่นั่งฟังอยู่แล้วพูดแทรกขึ้นมาขำๆ "
                    f"ห้ามแนะนำตัว ห้ามตอบยาว ห้ามเป็นทางการ]"
                )
            else:
                user_msg = text

            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_msg,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name
            )

        # Record bot reply in memory
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id="bot",
            sender_name=settings.bot_name,
            text=reply_text,
            is_bot=True
        )

        # Send reply back to LINE safely chunked within character limits
        text_messages = [TextMessage(text=c) for c in chunk_line_text(reply_text)]
        try:
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=text_messages
            )
            reply_res = messaging_api.reply_message(reply_request)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id)
            logger.info("Replied to [%s] in chat [%s] (spontaneous=%s)", sender_name, chat_id, is_spontaneous)
        except Exception as reply_err:
            logger.warning("reply_message failed (%s), attempting push_message fallback...", reply_err)
            if chat_id:
                try:
                    push_req = PushMessageRequest(to=chat_id, messages=text_messages)
                    push_res = messaging_api.push_message(push_req)
                    if push_res and hasattr(push_res, "sent_messages"):
                        for sm in push_res.sent_messages:
                            if hasattr(sm, "id") and sm.id:
                                memory_manager.register_bot_message_id(sm.id)
                    logger.info("Push fallback delivered successfully to [%s]", chat_id)
                except Exception as push_err:
                    logger.error("push_message fallback also failed: %s", push_err)


async def process_image_message(event: MessageEvent):
    """Handles an incoming image message event."""
    chat_id, user_id, is_group = extract_chat_and_user_ids(event)

    with get_api_client() as api_client:
        sender_name = await asyncio.to_thread(resolve_sender_name, api_client, chat_id, user_id, is_group)
        blob_api = MessagingApiBlob(api_client)
        messaging_api = MessagingApi(api_client)

        try:
            image_bytes = await asyncio.to_thread(blob_api.get_message_content, event.message.id)
        except Exception as e:
            logger.error("Failed to fetch image binary from LINE: %s", e)
            return

        current_trigger_mode = memory_manager.get_trigger_mode(chat_id, settings.group_trigger_mode)

        # In groups, check if we should spontaneously react or skip
        if is_group and current_trigger_mode != "all":
            mem = memory_manager.get_group_memory(chat_id)
            now = time.time()
            can_spontaneously_react = (
                current_trigger_mode in ("chime_in", "smart", "auto")
                and mem.messages_since_bot_spoke >= settings.spontaneous_min_messages
                and (mem.last_bot_reply_time == 0 or (now - mem.last_bot_reply_time) >= settings.spontaneous_cooldown_seconds)
                and random.random() < settings.spontaneous_image_rate
            )

            if not can_spontaneously_react:
                memory_manager.add_message(
                    chat_id=chat_id,
                    sender_id=user_id or "unknown",
                    sender_name=sender_name,
                    text="[ส่งรูปภาพ]",
                    is_bot=False,
                    image_desc="รูปภาพที่ส่งเข้ามาในแชท"
                )
                logger.info("Group image saved to history buffer without reply: [%s]", sender_name)
                return
            else:
                logger.info("Spontaneous image comment triggered in chat [%s]!", chat_id)

        # Show loading animation (1-on-1 only)
        if settings.enable_loading_animation and not is_group and user_id:
            try:
                anim_req = ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=15)
                messaging_api.show_loading_animation(anim_req)
            except Exception as e:
                logger.debug("Loading animation request ignored: %s", e)

        persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
        system_instruction = build_system_prompt(settings.bot_name, persona_key)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)

        user_prompt = "เพื่อนส่งรูปนี้มาในห้องไลน์ ช่วยดูรูปแล้วคอมเมนต์ แซว หรือพูดคุยสั้นๆ 1-2 ประโยค สไตล์เพื่อนสนิทในกลุ่มที่เป็นคนจริงๆ เป็นธรรมชาติ"

        reply_text = await asyncio.to_thread(
            gemini_client.generate_chat_response,
            user_message=user_prompt,
            system_instruction=system_instruction,
            history_context=history_context,
            sender_name=sender_name,
            image_bytes=image_bytes,
            mime_type="image/jpeg"
        )

        memory_manager.add_message(
            chat_id=chat_id,
            sender_id="bot",
            sender_name=settings.bot_name,
            text=reply_text,
            is_bot=True
        )

        text_messages = [TextMessage(text=c) for c in chunk_line_text(reply_text)]
        try:
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=text_messages
            )
            reply_res = messaging_api.reply_message(reply_request)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id)
            logger.info("Replied to image from [%s] in chat [%s]", sender_name, chat_id)
        except Exception as reply_err:
            logger.warning("reply_message for image failed (%s), attempting push fallback...", reply_err)
            if chat_id:
                try:
                    push_req = PushMessageRequest(to=chat_id, messages=text_messages)
                    messaging_api.push_message(push_req)
                except Exception as push_err:
                    logger.error("push_message fallback failed: %s", push_err)


async def process_join_event(event: JoinEvent):
    """Greets the group warmly when the bot is added to a LINE group or room."""
    chat_id, _, _ = extract_chat_and_user_ids(event)
    welcome_text = (
        f"โย่วทุกคน! 👋 ผม '{settings.bot_name}' สมาชิกใหม่สายซัพพอร์ตประจำกลุ่มนะครับ 555 "
        f"คุยเล่น ปรึกษา แซว ปลุกไฟ หรือวางแผนงานได้หมดเหมือนเพื่อนคนนึงเลย!\n\n"
        f"📌 วิธีเรียกผม:\n"
        f"- พิมพ์ชื่อ '{settings.bot_name}' หรือแท็ก @{settings.bot_name}\n"
        f"- ตอบกลับ (Quote Reply) ข้อความของผม\n"
        f"- คำสั่งเด็ด: `/plan` (วางแผนงาน), `/boost` (ปลุกพลังใจ), `/goal` (ตั้งเป้าหมาย), `/help`\n\n"
        f"ยินดีที่ได้รู้จักทุกคนนะเพื่อนๆ พร้อมลุย! 🚀"
    )
    with get_api_client() as api_client:
        messaging_api = MessagingApi(api_client)
        try:
            req = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=welcome_text)]
            )
            reply_res = messaging_api.reply_message(req)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id)
            logger.info("Sent join welcome message to chat [%s]", chat_id)
        except Exception as e:
            logger.error("Failed to send join greeting: %s", e)


async def process_follow_event(event: FollowEvent):
    """Greets a user when they add the bot in a 1-on-1 private chat."""
    welcome_text = (
        f"หวัดดีครับ! 👋 ผม '{settings.bot_name}' เพื่อนซี้ AI ประจำตัวคุณนะ 555\n"
        f"คุยเล่น ปรึกษาปัญหาชีวิต วางแผนงาน หรือตั้งเป้าหมายได้ตลอดเวลาเลย\n\n"
        f"💡 ลองพิมพ์คุยเล่น หรือใช้คำสั่ง:\n"
        f"- `/plan [เรื่อง]` : วางแผนกลยุทธ์แบบ Step-by-Step\n"
        f"- `/boost [เรื่อง]` : ปลุกพลังใจและวิธีทะลวงจุดตัน\n"
        f"- `/goal [เรื่อง]` : แปลงเป้าหมายเป็น SMART Goal\n"
        f"- `/help` : ดูคำสั่งทั้งหมด\n\n"
        f"หรือดึงผมเข้ากลุ่มส่วนตัวไว้คุยกับแก๊งเพื่อนก็ได้นะ! 🚀"
    )
    with get_api_client() as api_client:
        messaging_api = MessagingApi(api_client)
        try:
            req = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=welcome_text)]
            )
            reply_res = messaging_api.reply_message(req)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id)
        except Exception as e:
            logger.error("Failed to send follow greeting: %s", e)
