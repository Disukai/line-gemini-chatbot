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
    FileMessageContent,
    JoinEvent,
    FollowEvent,
)

from app.config import settings
from app.memory import memory_manager
from app.persona import build_system_prompt
from app.commands import parse_command, execute_command
from app.gemini_client import gemini_client
from app.chat_tracker import chat_tracker
from app.file_utils import (
    get_file_category,
    decode_text_file,
    extract_text_from_docx,
    extract_text_from_xlsx,
)

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
    chat_tracker.register_chat(chat_id, "group" if is_group else "user")

    message_id = getattr(event.message, "id", None)
    quoted_id = getattr(event.message, "quoted_message_id", None)

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

        # Record incoming message in memory buffer with message_id and quoted_message_id
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id=user_id or "unknown",
            sender_name=sender_name,
            text=text,
            is_bot=False,
            message_id=message_id,
            quoted_message_id=quoted_id
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

        # Look up replied/quoted message details if user quoted someone
        quoted_msg = memory_manager.get_message(chat_id, quoted_id) if quoted_id else None
        quoted_context_block = ""
        if quoted_msg:
            q_author = f"{settings.bot_name} (ตัวคุณเอง)" if quoted_msg.is_bot else quoted_msg.sender_name
            extras = []
            if quoted_msg.image_desc:
                extras.append(f"รูปภาพ: {quoted_msg.image_desc}")
            if quoted_msg.file_desc:
                extras.append(f"ไฟล์แนบ: {quoted_msg.file_desc}")
            extra_info = f" [{', '.join(extras)}]" if extras else ""

            quoted_context_block = (
                f"[บริบทสำคัญ: {sender_name} กำลังกดรีพลาย (Quote Reply) ตอบกลับข้อความเดิมของ '{q_author}']\n"
                f"- ข้อความเดิมที่ถูกรีพลาย: \"{quoted_msg.text}\"{extra_info}\n"
                f"- ข้อความใหม่ที่ {sender_name} พิมพ์ตอบกลับมา: \"{text}\"\n"
                f"คำแนะนำสำหรับ {settings.bot_name}: ให้ตอบโดยเชื่อมโยงกับข้อความเดิมที่ {sender_name} ตอบกลับมาอย่างเป็นธรรมชาติ "
                f"เหมือนเพื่อนที่จำได้ว่ากำลังคุยเรื่องอะไรกันอยู่ ไม่ต้องพูดซ้ำประโยคเดิมหมด แค่คุยต่อให้ลื่นไหล\n\n"
            )
        elif quoted_id:
            quoted_context_block = (
                f"[บริบท: {sender_name} กำลังกดรีพลาย (Quote Reply) ตอบกลับข้อความเดิมในแชท แต่ข้อความนั้นเก่าเกินกว่าประวัติในความจำ]\n"
                f"- ข้อความที่ {sender_name} พิมพ์: \"{text}\"\n\n"
            )

        # Check for slash commands
        parsed_cmd = parse_command(text)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)

        if parsed_cmd:
            cmd, args = parsed_cmd
            # If user ran a command like /plan or /boost without args but quoted a message, use the quoted message text!
            if not args.strip() and quoted_msg and quoted_msg.text:
                args = quoted_msg.text
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
                    f"{quoted_context_block}{text}\n\n"
                    f"[บรีฟสำหรับ {settings.bot_name}: เพื่อนกำลังคุยกันในกลุ่ม ให้ตอบแจมหรือแซวสั้นๆ 1-2 ประโยค "
                    f"ฟีลเพื่อน Gen Z / Gen Alpha ในกลุ่มนั่งฟังอยู่แล้วสวนกลับมาแบบกวนๆ หรือช็อตฟีลขำๆ "
                    f"ใช้ภาษาแชทวัยรุ่นไทยสมัยนี้ (นอย, อ่อม, ทำถึง, ฉ่ำ, ช็อตฟีล, เกิ๊น, ตัวแม่, ของแทร่, 5555555) "
                    f"ห้ามใช้สำนวนแปลหนังฝรั่งหรือนิยายเด็ดขาด ห้ามแนะนำตัว]"
                )
            else:
                user_msg = f"{quoted_context_block}{text}"

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
                        memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
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
                                memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
                    logger.info("Push fallback delivered successfully to [%s]", chat_id)
                except Exception as push_err:
                    logger.error("push_message fallback also failed: %s", push_err)


async def process_image_message(event: MessageEvent):
    """
    Handles an incoming image message event.
    Performs OCR, document/receipt analysis, homework/code solving, or witty Gen Z photo commentary.
    Supports quote-reply context if sent in response to another message.
    """
    chat_id, user_id, is_group = extract_chat_and_user_ids(event)
    chat_tracker.register_chat(chat_id, "group" if is_group else "user")

    image_id = getattr(event.message, "id", None)
    quoted_id = getattr(event.message, "quoted_message_id", None)

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

        # Look up replied message if this image was sent as a quote reply
        quoted_msg = memory_manager.get_message(chat_id, quoted_id) if quoted_id else None
        quoted_image_context = ""
        if quoted_msg:
            q_author = f"{settings.bot_name} (ตัวคุณเอง)" if quoted_msg.is_bot else quoted_msg.sender_name
            quoted_image_context = (
                f"\n[บริบทเพิ่มเติม: ผู้ใช้ส่งรูปภาพนี้มาเพื่อตอบกลับ (Quote Reply) ข้อความของ '{q_author}': \"{quoted_msg.text}\"]\n"
            )

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

            # If user quoted bot's message directly, always react
            if quoted_id and memory_manager.is_bot_message(quoted_id):
                can_spontaneously_react = True

            if not can_spontaneously_react:
                memory_manager.add_message(
                    chat_id=chat_id,
                    sender_id=user_id or "unknown",
                    sender_name=sender_name,
                    text="[ส่งรูปภาพ]",
                    is_bot=False,
                    message_id=image_id,
                    quoted_message_id=quoted_id,
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

        user_prompt = (
            f"เพื่อนส่งรูปภาพนี้มาในแชท {quoted_image_context}ช่วยดูและอ่านรายละเอียดในรูปภาพอย่างละเอียด:\n"
            "1. OCR & อ่านข้อความ: ถ้าในรูปมีตัวหนังสือ ป้าย ข้อความ สลิป ใบเสร็จ เอกสาร การบ้าน เมนูอาหาร สกรีนช็อตโค้ด หรือหน้าจอแชท ให้อ่านข้อความทั้งหมดและช่วยแปล/ตอบ/วิเคราะห์/สรุป/ช่วยคิดเงินให้เพื่อนทันที\n"
            "2. ถ้าเป็นคำถามหรือการบ้าน: ช่วยตอบและอธิบายเฉลยให้ถูกต้อง ชัดเจน\n"
            "3. ถ้าเป็นรูปทั่วไป/มีม/สถานที่/ของกิน/รูปคน: สังเกตดีเทลในรูปแล้วคุย แซว หรือเม้าท์แบบเพื่อนซี้ Gen Z รู้จริง ไม่พูดลอยๆ (ช็อตฟีล, ป้ายยา, แซวดีเทลในรูป)\n"
            "4. โทนการพูด: เพื่อนสนิท Gen Z / Gen Alpha สมัยนี้ (ทำถึง, โฮ่งมาก, นอย, ฉ่ำ, ติดแกลม, สภาพพพ, 5555555) ห้ามสำนวนแปลนิยาย/หนังฝรั่งเด็ดขาด ห้ามทักทายแบบทางการ"
        )

        reply_text = await asyncio.to_thread(
            gemini_client.generate_chat_response,
            user_message=user_prompt,
            system_instruction=system_instruction,
            history_context=history_context,
            sender_name=sender_name,
            image_bytes=image_bytes,
            mime_type="image/jpeg"
        )

        # Save incoming image action and bot reply into memory buffer
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id=user_id or "unknown",
            sender_name=sender_name,
            text="[ส่งรูปภาพ]",
            is_bot=False,
            message_id=image_id,
            quoted_message_id=quoted_id,
            image_desc="รูปภาพ/เอกสาร/สลิปที่ส่งเข้ามาในแชท"
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
                        memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
            logger.info("Replied to image from [%s] in chat [%s]", sender_name, chat_id)
        except Exception as reply_err:
            logger.warning("reply_message for image failed (%s), attempting push fallback...", reply_err)
            if chat_id:
                try:
                    push_req = PushMessageRequest(to=chat_id, messages=text_messages)
                    messaging_api.push_message(push_req)
                except Exception as push_err:
                    logger.error("push_message fallback failed: %s", push_err)


async def process_file_message(event: MessageEvent):
    """
    Handles incoming document/file uploads (PDF, Word docx, Excel xlsx, CSV, TXT, Code, Audio, etc.).
    Extracts text or attaches native media parts for deep analysis by Gemini.
    Supports quote-reply context if uploaded in response to another message.
    """
    chat_id, user_id, is_group = extract_chat_and_user_ids(event)
    chat_tracker.register_chat(chat_id, "group" if is_group else "user")

    file_name = getattr(event.message, "file_name", "document")
    file_size = getattr(event.message, "file_size", 0)
    file_id = event.message.id
    quoted_id = getattr(event.message, "quoted_message_id", None)

    with get_api_client() as api_client:
        sender_name = await asyncio.to_thread(resolve_sender_name, api_client, chat_id, user_id, is_group)
        blob_api = MessagingApiBlob(api_client)
        messaging_api = MessagingApi(api_client)

        try:
            file_bytes = await asyncio.to_thread(blob_api.get_message_content, file_id)
        except Exception as e:
            logger.error("Failed to fetch file binary from LINE: %s", e)
            return

        # Show loading animation in 1-on-1 chat
        if settings.enable_loading_animation and not is_group and user_id:
            try:
                anim_req = ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=20)
                messaging_api.show_loading_animation(anim_req)
            except Exception as e:
                logger.debug("Loading animation request ignored: %s", e)

        category, mime_type = get_file_category(file_name)
        logger.info(
            "Processing file [%s] (size=%d bytes, category=%s, mime=%s) from [%s] in [%s]",
            file_name, file_size, category, mime_type, sender_name, chat_id
        )

        # Look up replied message if this file was sent as a quote reply
        quoted_msg = memory_manager.get_message(chat_id, quoted_id) if quoted_id else None
        quoted_file_context = ""
        if quoted_msg:
            q_author = f"{settings.bot_name} (ตัวคุณเอง)" if quoted_msg.is_bot else quoted_msg.sender_name
            quoted_file_context = (
                f"\n[บริบทเพิ่มเติม: ผู้ใช้ส่งไฟล์นี้มาเพื่อตอบกลับ (Quote Reply) ข้อความของ '{q_author}': \"{quoted_msg.text}\"]\n"
            )

        persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
        system_instruction = build_system_prompt(settings.bot_name, persona_key)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)

        reply_text = ""

        if category == "pdf":
            user_prompt = (
                f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ PDF ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                f"ช่วยอ่านเนื้อหาในไฟล์นี้ทั้งหมดอย่างละเอียด แล้วสรุปใจความสำคัญ ประเด็นหลัก หรือสิ่งที่น่าสนใจให้ฟัง\n"
                f"- ถ้ามี Action Items, ข้อตกลง, หรือตัวเลขสำคัญให้ดึงมาบอก\n"
                f"- ตอบด้วยภาษาเพื่อนสนิท Gen Z / Gen Alpha (สรุปให้ฉ่ำๆ, ทำถึง, เข้าใจง่าย, มี bullet points อ่านสบายตา)\n"
                f"- ห้ามภาษาทางการแข็งทื่อ ห้ามสำนวนนิยาย"
            )
            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_prompt,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name,
                media_bytes=file_bytes,
                media_mime_type="application/pdf"
            )

        elif category == "docx":
            text_content = extract_text_from_docx(file_bytes)
            if not text_content.strip():
                reply_text = f"แกรรร เราลองเปิดอ่านไฟล์ Word '{file_name}' แล้วแต่ดูเหมือนไฟล์จะว่างเปล่าหรือเป็นรูปล้วนๆ เลยอ่านข้อความข้างในไม่เจออะ 5555555"
            else:
                user_prompt = (
                    f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ Word (.docx) ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                    f"เนื้อหาในเอกสาร:\n"
                    f"\"\"\"\n{text_content[:40000]}\n\"\"\"\n\n"
                    f"ช่วยอ่านเนื้อหาในเอกสารนี้ แล้วสรุปประเด็นสำคัญ สาระสำคัญ หรืออธิบายสิ่งที่อยู่ในไฟล์ให้เพื่อนฟังแบบเข้าใจง่ายๆ\n"
                    f"สไตล์เพื่อน Gen Z (สรุปแบบทำถึง, สรุปให้ฉ่ำ, อ่านง่าย) ห้ามใช้ภาษาทางการแข็งทื่อ"
                )
                reply_text = await asyncio.to_thread(
                    gemini_client.generate_chat_response,
                    user_message=user_prompt,
                    system_instruction=system_instruction,
                    history_context=history_context,
                    sender_name=sender_name
                )

        elif category == "xlsx":
            sheet_content = extract_text_from_xlsx(file_bytes)
            if not sheet_content.strip():
                reply_text = f"แกรรร ไฟล์ Excel '{file_name}' นี้ดูเหมือนไม่มีข้อมูลตัวอักษรหรือตารางที่เราอ่านได้เลย ลองเซฟเป็น CSV หรือส่งเป็นรูปตารางมาดูมั้ยย 5555555"
            else:
                user_prompt = (
                    f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ตาราง Excel (.xlsx) ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                    f"ข้อมูลตารางในไฟล์:\n"
                    f"\"\"\"\n{sheet_content[:40000]}\n\"\"\"\n\n"
                    f"ช่วยวิเคราะห์และสรุปข้อมูลในตารางนี้ให้เพื่อนฟัง ดึง insight สำคัญ ยอดรวม หรือประเด็นเด่นๆ ออกมาสรุปให้ชัดเจน\n"
                    f"สไตล์เพื่อน Gen Z ฉลาดๆ สรุปแบบทำถึง อ่านง่าย ห้ามตอบทางการน่าเบื่อ"
                )
                reply_text = await asyncio.to_thread(
                    gemini_client.generate_chat_response,
                    user_message=user_prompt,
                    system_instruction=system_instruction,
                    history_context=history_context,
                    sender_name=sender_name
                )

        elif category == "text":
            text_content = decode_text_file(file_bytes)
            user_prompt = (
                f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ '{file_name}' มาในแชท {quoted_file_context}\n"
                f"เนื้อหาในไฟล์:\n"
                f"\"\"\"\n{text_content[:40000]}\n\"\"\"\n\n"
                f"ช่วยอ่านโค้ด/ข้อมูล/ข้อความในไฟล์นี้ แล้วสรุป อธิบาย หรือวิเคราะห์ให้เพื่อนฟัง\n"
                f"- ถ้าเป็นโค้ด: อธิบายการทำงาน และแนะนำจุดเด่นหรือจุดปรับปรุง/บั๊กถ้ามี\n"
                f"- ถ้าเป็นข้อมูล/CSV/JSON: สรุปข้อมูลสำคัญและ insight ที่น่าสนใจ\n"
                f"- ถ้าเป็นบทความ/บันทึก: สรุปใจความสำคัญ\n"
                f"ตอบด้วยภาษาเพื่อนสนิท Gen Z ที่ฉลาดและเข้าใจง่าย ห้ามทางการ ห้ามสำนวนนิยาย"
            )
            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_prompt,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name
            )

        elif category == "image":
            user_prompt = (
                f"เพื่อนส่งไฟล์รูปภาพชื่อ '{file_name}' มาในแชท {quoted_file_context}ช่วยอ่านและดูรายละเอียดในรูปอย่างละเอียด:\n"
                f"1. OCR & อ่านข้อความทั้งหมดในรูป (ถ้าเป็นบิล/สลิป/เอกสาร/การบ้าน/โค้ด)\n"
                f"2. สรุป ตอบคำถาม หรือวิเคราะห์สิ่งที่เห็น\n"
                f"3. ตอบสไตล์เพื่อนซี้ Gen Z รู้จริง ทำถึง ห้ามสำนวนนิยาย"
            )
            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_prompt,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name,
                media_bytes=file_bytes,
                media_mime_type=mime_type
            )

        elif category == "audio":
            user_prompt = (
                f"เพื่อนส่งไฟล์เสียงชื่อ '{file_name}' มาในแชท {quoted_file_context}ช่วยฟังเนื้อหาในคลิปเสียงนี้อย่างละเอียด "
                f"แล้วถอดความหรือสรุปใจความสำคัญให้เพื่อนฟัง ตอบด้วยสไตล์เพื่อน Gen Z สรุปกระชับเข้าใจง่าย"
            )
            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_prompt,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name,
                media_bytes=file_bytes,
                media_mime_type=mime_type
            )

        else:
            size_kb = max(1, file_size // 1024)
            reply_text = (
                f"แกรรร ไฟล์ '{file_name}' ({size_kb} KB) อันนี้เรายังแกะเนื้อหาข้างในไม่ได้อะ 5555555\n"
                f"ตอนนี้เรารองรับอ่านไฟล์ PDF, Word (.docx), Excel (.xlsx), CSV, Text/Code (.txt, .json, .py, .md ฯลฯ), รูปภาพ และไฟล์เสียง นะแกกก "
                f"ลองแปลงไฟล์หรือแคปรูปส่งมาให้เราดูใหม่อีกทีนะะะ!"
            )

        # Record file message and bot reply in conversation history
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id=user_id or "unknown",
            sender_name=sender_name,
            text=f"[ส่งไฟล์: {file_name}]",
            is_bot=False,
            message_id=file_id,
            quoted_message_id=quoted_id,
            file_desc=f"{file_name} ({file_size} bytes)"
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
                        memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
            logger.info("Replied to file [%s] from [%s] in chat [%s]", file_name, sender_name, chat_id)
        except Exception as reply_err:
            logger.warning("reply_message for file failed (%s), attempting push fallback...", reply_err)
            if chat_id:
                try:
                    push_req = PushMessageRequest(to=chat_id, messages=text_messages)
                    push_res = messaging_api.push_message(push_req)
                    if push_res and hasattr(push_res, "sent_messages"):
                        for sm in push_res.sent_messages:
                            if hasattr(sm, "id") and sm.id:
                                memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
                except Exception as push_err:
                    logger.error("push_message fallback failed: %s", push_err)


async def process_join_event(event: JoinEvent):
    """Greets the group warmly when the bot is added to a LINE group or room."""
    chat_id, _, _ = extract_chat_and_user_ids(event)
    chat_tracker.register_chat(chat_id, "group")
    welcome_text = (
        f"ดีค่าา / ดีครับทุกคนนน! โทมัส ({settings.bot_name}) รายงานตัววว 5555555 "
        f"เข้ามาช่วยปั่น ช่วยเม้าท์ ชวนคุยในกลุ่มละนะแกรรร ใครมีอะไรให้ช่วยคิด วางแผนงาน แซวเพื่อน หรืออยากได้คนร่วมวงเม้าท์ เรียกชั้นได้ตลอดเลยยย\n\n"
        f"📌 วิธีเรียกชั้น:\n"
        f"- พิมพ์ชื่อ '{settings.bot_name}' หรือแท็ก @{settings.bot_name}\n"
        f"- ตอบกลับ (Quote Reply) ข้อความของชั้น\n"
        f"- คำสั่งเด็ด: `/news` (สรุปข่าวดังวันนี้แบบทำถึง), `/plan` (วางแผนงานแบบตัวแม่), `/boost`, `/help`\n\n"
        f"พร้อมเปิดตี้ละพวกแกรรร ลุยยยย 🔥"
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
        f"หวัดดีแกรรร! 👋 เรา '{settings.bot_name}' เพื่อนซี้ AI ประจำตัวแกเอง 5555555\n"
        f"มีอะไรมาเม้าท์ มาปรึกษา วางแผนงาน หรือบ่นชีวิตได้ตลอดเวลาเลยนะ ฟีลเพื่อนสนิทคุยกันชิลๆ\n\n"
        f"💡 ลองพิมพ์คุยเล่น หรือเล่นคำสั่งจึ้งๆ:\n"
        f"- `/news` : สรุปข่าวดังวันนี้แบบฉ่ำๆ ไม่ตกเทรนด์\n"
        f"- `/plan [เรื่อง]` : วางแผนกลยุทธ์แบบทำถึง Step-by-Step\n"
        f"- `/boost [เรื่อง]` : ปลุกพลังใจ บูสต์เอเนอร์จี้เวลาหมดไฟ\n"
        f"- `/goal [เรื่อง]` : ตั้งเป้าหมาย SMART Goal ชัดเจน\n"
        f"- `/help` : ดูคำสั่งทั้งหมด\n\n"
        f"หรือดึงเราเข้ากลุ่มไลน์ไปป่วนแก๊งเพื่อนก็ได้นะแกกก! 🚀"
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
