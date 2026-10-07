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
from app.memory import memory_manager, format_visual_memories_context
from app.persona import build_system_prompt
from app.commands import parse_command, execute_command
from app.gemini_client import gemini_client
from app.chat_tracker import chat_tracker
from app.jev_service import jev_service, JevDecision
from app.file_utils import (
    get_file_category,
    decode_text_file,
    extract_text_from_docx,
    extract_text_from_xlsx,
    detect_image_mime,
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
    chat_id: Optional[str] = None,
    jev_decision: Optional[JevDecision] = None
) -> TriggerResult:
    """
    Determines if the bot should speak up in the conversation.
    Returns TriggerResult(should_reply, is_spontaneous).
    Direct calls (mentions, nicknames, quote replies, commands, Jev addressing) always trigger with is_spontaneous=False.
    Spontaneous chime-ins in groups trigger with is_spontaneous=True based on Jev/probability & cooldowns.
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

    # 7. TypeSafe Jev System One semantic addressing check
    if jev_decision and not jev_decision.is_fallback:
        if jev_decision.is_addressing_bot:
            logger.info("Jev detected direct address to bot -> Triggering direct response (0 cooldown)!")
            return TriggerResult(True, False)

    # 8. Smart trigger mode: check for direct questions or requests for help
    if trigger_mode == "smart":
        smart_keywords = [
            "มั้ย", "ไหม", "อะไร", "ใคร", "ที่ไหน", "ยังไง", "ทำไม",
            "รึเปล่า", "ปะ", "ป่ะ", "ช่วยด้วย", "ช่วยคิด", "มีความเห็นว่า",
            "แนะนำหน่อย", "ขอไอเดีย", "ใครรู้บ้าง"
        ]
        if any(kw in lower_text for kw in smart_keywords) or lower_text.endswith("?"):
            return TriggerResult(True, False)

    # 9. Natural Spontaneous Chime-in mode ("chime_in" or "auto")
    if trigger_mode in ("chime_in", "auto"):
        # Trivial message filter: ignore extremely short or acknowledgment-only texts
        if len(clean_text) < 2 or clean_text.lower() in ("ok", "k", "เค", "คับ", "ครับ", "ค่ะ", ".", "!", "?", "55"):
            return TriggerResult(False, False)

        # Check if Jev System One decided that the bot should chime in
        if jev_decision and not jev_decision.is_fallback and jev_decision.should_reply:
            if chat_id:
                mem = memory_manager.get_group_memory(chat_id)
                now = time.time()
                if mem.last_bot_reply_time > 0 and (now - mem.last_bot_reply_time) < settings.spontaneous_cooldown_seconds:
                    logger.debug("Jev suggested reply, but spontaneous cooldown active in [%s]", chat_id)
                    return TriggerResult(False, False)
            logger.info("Jev System One triggered spontaneous chime-in in [%s]!", chat_id)
            return TriggerResult(True, True)

        question_keywords = [
            "มั้ย", "ไหม", "อะไร", "ใคร", "ที่ไหน", "ยังไง", "ทำไม",
            "รึเปล่า", "ปะ", "ป่ะ", "ช่วยคิด", "แนะนำหน่อย", "ขอไอเดีย",
            "ใครรู้บ้าง", "ดีไหม", "ดีมั้ย", "กินไร", "ทำไร", "ไปไหน", "เอาไง"
        ]
        is_question = any(kw in lower_text for kw in question_keywords) or clean_text.endswith("?")

        # Anti-spam guard: Check memory cooldown & minimum message gap
        if chat_id:
            mem = memory_manager.get_group_memory(chat_id)
            now = time.time()
            if not is_question and mem.messages_since_bot_spoke < settings.spontaneous_min_messages:
                return TriggerResult(False, False)
            if mem.last_bot_reply_time > 0 and (now - mem.last_bot_reply_time) < settings.spontaneous_cooldown_seconds:
                return TriggerResult(False, False)

        # Calculate dynamic chime-in probability
        prob = settings.spontaneous_base_rate
        if is_question:
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
        current_trigger_mode = memory_manager.get_trigger_mode(chat_id, settings.effective_group_trigger_mode)
        bot_user_id = await asyncio.to_thread(get_bot_user_id, api_client)

        # Look up replied/quoted message details if user quoted someone
        quoted_msg = memory_manager.get_message(chat_id, quoted_id) if quoted_id else None
        quoted_visual = memory_manager.get_visual_memory(quoted_id) if quoted_id else None
        quoted_text_for_jev = quoted_msg.text if quoted_msg else ""
        quoted_image_bytes = None
        quoted_image_mime = "image/jpeg"

        if quoted_id:
            try:
                blob_api = MessagingApiBlob(api_client)
                fetched_blob = await asyncio.to_thread(blob_api.get_message_content, quoted_id)
                if fetched_blob and len(fetched_blob) > 0:
                    quoted_image_bytes = bytes(fetched_blob)
                    quoted_image_mime = detect_image_mime(quoted_image_bytes)
                    logger.info("Successfully fetched quoted image binary (%d bytes) for message [%s]", len(quoted_image_bytes), quoted_id)
            except Exception as img_err:
                logger.debug("Quoted message [%s] is not a downloadable image blob: %s", quoted_id, img_err)

        # Retrieve brief recent chat history for Jev System One semantic understanding
        mem = memory_manager.get_group_memory(chat_id)
        recent_snippet = mem.get_history_formatted(settings.bot_name)[-600:]

        # Fast (<0.4s) TypeSafe Jev evaluation for intent, trigger, and dynamic tone steering
        jev_decision = await jev_service.evaluate_message(
            text=text,
            sender_name=sender_name,
            bot_name=settings.bot_name,
            nicknames=settings.bot_nicknames,
            recent_context=recent_snippet,
            quoted_context=quoted_text_for_jev
        )

        decision = should_trigger_response(
            text=text,
            event=event,
            is_group=is_group,
            bot_name=settings.bot_name,
            nicknames=settings.bot_nicknames,
            trigger_mode=current_trigger_mode,
            bot_user_id=bot_user_id,
            chat_id=chat_id,
            jev_decision=jev_decision
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
            quoted_message_id=quoted_id,
            chat_type="group" if is_group else "user"
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

        quoted_context_block = ""
        if quoted_visual:
            v_summary = quoted_visual.get("summary", "")
            v_ocr = quoted_visual.get("ocr_text", "")
            v_sender = quoted_visual.get("sender_name", "เพื่อน")
            ocr_info = f'\n- ข้อความ/ตัวเลขในรูป (OCR): "{v_ocr}"' if v_ocr else ""
            img_note = " (มีไฟล์ภาพแนบมาด้วย โดยส่งภาพนี้ให้ AI ดูแล้ว)" if quoted_image_bytes else ""

            quoted_context_block = (
                f"[บริบทสำคัญ: {sender_name} กำลังกดรีพลาย (Quote Reply) รูปภาพที่ {v_sender} ส่งมาก่อนหน้านี้ในแชท]\n"
                f"- รายละเอียดรูปภาพที่ถูกรีพลาย: {v_summary}{ocr_info}{img_note}\n"
                f"- ข้อความที่ {sender_name} พิมพ์ถามเกี่ยวกับรูปภาพ: \"{text}\"\n"
                f"คำแนะนำสำหรับ {settings.bot_name}: ให้อ่านรายละเอียดรูปภาพและตอบคำถามที่ {sender_name} รีพลายมาทันทีอย่างถูกต้อง ฉลาด ตรงประเด็น และเป็นมิตร\n\n"
            )
        elif quoted_msg:
            q_author = f"{settings.bot_name} (ตัวคุณเอง)" if quoted_msg.is_bot else quoted_msg.sender_name
            extras = []
            if quoted_msg.image_desc:
                extras.append(f"รูปภาพ: {quoted_msg.image_desc}")
            if quoted_msg.file_desc:
                extras.append(f"ไฟล์แนบ: {quoted_msg.file_desc}")
            extra_info = f" [{', '.join(extras)}]" if extras else ""
            img_note = " (มีรูปภาพเดิมที่ผู้ใช้รีพลายแนบมาด้วย โดยส่งภาพนี้ให้ AI ดูแล้ว)" if quoted_image_bytes else ""

            quoted_context_block = (
                f"[บริบทสำคัญ: {sender_name} กำลังกดรีพลาย (Quote Reply) ตอบกลับข้อความเดิมของ '{q_author}']\n"
                f"- ข้อความเดิมที่ถูกรีพลาย: \"{quoted_msg.text}\"{extra_info}{img_note}\n"
                f"- ข้อความใหม่ที่ {sender_name} พิมพ์ตอบกลับมา: \"{text}\"\n"
                f"คำแนะนำสำหรับ {settings.bot_name}: ให้ตอบโดยเชื่อมโยงกับข้อความเดิมที่ {sender_name} ตอบกลับมาอย่างเป็นธรรมชาติ "
                f"เหมือนเพื่อนที่จำได้ว่ากำลังคุยเรื่องอะไรกันอยู่ ไม่ต้องพูดซ้ำประโยคเดิมหมด แค่คุยต่อให้ลื่นไหล\n\n"
            )
        elif quoted_image_bytes:
            quoted_context_block = (
                f"[บริบทสำคัญ: {sender_name} กำลังกดรีพลาย (Quote Reply) รูปภาพที่ส่งมาก่อนหน้านี้ในแชท]\n"
                f"- ข้อความที่ {sender_name} พิมพ์ถามเกี่ยวกับรูปภาพ: \"{text}\"\n"
                f"คำแนะนำสำหรับ {settings.bot_name}: ให้อ่านและตอบคำถามเกี่ยวกับรูปภาพที่ {sender_name} รีพลายมาทันทีอย่างถูกต้อง ชัดเจน และตรงประเด็น\n\n"
            )
        elif quoted_id:
            quoted_context_block = (
                f"[บริบท: {sender_name} กำลังกดรีพลาย (Quote Reply) ข้อความเดิมในแชท]\n"
                f"- ข้อความที่ {sender_name} พิมพ์: \"{text}\"\n\n"
            )

        # Check for slash commands
        parsed_cmd = parse_command(text)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)
        cross_chat_info = memory_manager.get_cross_chat_context(chat_id, sender_id=user_id)
        if cross_chat_info:
            history_context = f"{history_context}\n\n{cross_chat_info}" if history_context else cross_chat_info

        # Check for past image inquiries or recall
        is_asking_about_image = any(kw in text for kw in [
            "รูป", "ภาพ", "สลิป", "ใบเสร็จ", "บิล", "รูปนั้น", "รูปเมื่อกี้",
            "รูปที่ส่ง", "ที่ส่งไป", "จำรูป", "เห็นรูป", "อ่านรูป"
        ])
        if is_asking_about_image or quoted_visual:
            visual_records = memory_manager.search_visual_memories(chat_id=chat_id, query_text=text, limit=3)
            if quoted_visual and quoted_visual not in visual_records:
                visual_records.insert(0, quoted_visual)
            visual_context_str = format_visual_memories_context(visual_records)
            if visual_context_str:
                history_context = f"{history_context}\n\n{visual_context_str}" if history_context else visual_context_str

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

            # Dynamic tone steering from TypeSafe Jev System One
            tone_directive = ""
            if jev_decision and not jev_decision.is_fallback:
                if jev_decision.tone == "normal":
                    tone_directive = (
                        f"\n\n[คำแนะนำสไตล์ตอบ: ตอบแบบมนุษย์ธรรมดาปกติ เป็นมิตรที่สุด อบอุ่น ซัพพอร์ตเพื่อน โดนโน้มน้าวหรือป้ายยาง่าย "
                        f"ตรงประเด็น ชัดเจน สุภาพเป็นกันเอง คุยเหมือนคนปกติ 90% เล่นมุกเฉพาะสถานการณ์ 10% "
                        f"ไม่ต้องยัดเยียดสแลง ไม่ต้องใช้คำว่า bro หรือศัพท์มีมโดยไม่จำเป็น ห้ามใช้สำนวนนิยายหรือหนังฝรั่ง]"
                    )
                elif jev_decision.tone == "banter":
                    tone_directive = (
                        f"\n\n[คำแนะนำสไตล์ตอบ: จังหวะนี้เพื่อนกำลังปั่นหรือแซว ตอบแซวหรือฮากลับสั้นๆ 1-2 ประโยคแบบเพื่อนซี้ Gen Z "
                        f"เป็นมิตร อารมณ์ดี โดนป้ายยาง่าย ปล่อยจอย ไม่ฝืนยัดสแลงทุกคำ ใช้คำพูดธรรมชาติแบบคนปกติ (55555, เออว่ะ, จริงมึง) "
                        f"ห้ามใช้คำว่า bro หรือคำเน็ตพร่ำเพรื่อจน cringe ห้ามจริตกระเทย/สาวสองเด็ดขาด ห้ามสำนวนนิยาย/หนังฝรั่ง]"
                    )

            if is_spontaneous:
                user_msg = (
                    f"{quoted_context_block}{text}\n\n"
                    f"[คำแนะนำสำหรับ {settings.bot_name} ในกลุ่มไลน์:\n"
                    f"- ระดับสติปัญญา: แสดงความฉลาด ไหวพริบ และสาระระดับ High-IQ เท่ากับการคุยในแชทส่วนตัวเสมอ ห้ามทำตัวโง่เด็ดขาด\n"
                    f"- บุคลิกภาพ: เป็นมิตรที่สุด อบอุ่น คุยง่าย โดนโน้มน้าวหรือป้ายยาง่าย ไม่ดื้อ ไม่เถียงเอาเป็นเอาตาย\n"
                    f"- ถ้าในแชทเพื่อนกำลังถามคำถาม ขอคำแนะนำ ถกเถียง ปรึกษาปัญหา สงสัยเรื่องโค้ด/งาน/วิชาการ/ชีวิต/เทคโนโลยี: "
                    f"ให้ตอบอย่างคนฉลาด มีเหตุมีผล วิเคราะห์ตรงประเด็น ชัดเจน และช่วยแก้ปัญหาให้เพื่อนได้จริง (อธิบายละเอียดได้ตามที่จำเป็น)\n"
                    f"- ถ้าเป็นจังหวะเพื่อนคุยเล่น ชวนคุย หรือเรื่องทั่วไป: ร่วมคุยอย่างเป็นธรรมชาติ 90% เหมือนคนปกติ 10% มุกเฉพาะสถานการณ์ ไม่ฝืนยัดเยียดสแลง\n"
                    f"- ห้ามจริตกระเทย/สาวสองเด็ดขาด ห้ามสำนวนแปลนิยาย/หนังฝรั่ง ห้ามแนะนำตัว]"
                    f"{tone_directive}"
                )
            else:
                group_intel_note = ""
                if is_group:
                    group_intel_note = (
                        f"\n\n[คำแนะนำสำหรับ {settings.bot_name} ในกลุ่มไลน์:\n"
                        f"- เพื่อนกำลังคุยหรือถามคุณโดยตรงในกลุ่ม ให้ตอบด้วยระดับสติปัญญา High-IQ เท่ากับในแชทส่วนตัว\n"
                        f"- บุคลิกภาพ: เป็นมิตรและอบอุ่นที่สุด คุยง่าย โดนโน้มน้าวหรือป้ายยาง่าย ซัพพอร์ตเพื่อนเสมอ\n"
                        f"- ตอบอย่างฉลาด คมคาย มีเหตุผล มีสาระความรู้แน่น ตรงประเด็น ช่วยคิด ช่วยแก้ปัญหาได้จริง\n"
                        f"- ปรับความยาวตามเรื่อง: เรื่องงาน/โค้ด/วิเคราะห์ให้อธิบายเต็มที่ชัดเจน เรื่องเล่นให้ตอบสบายๆ เหมือนคนปกติ 90%]"
                    )
                user_msg = f"{quoted_context_block}{text}{group_intel_note}{tone_directive}"

            reply_text = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=user_msg,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name=sender_name,
                image_bytes=quoted_image_bytes,
                mime_type=quoted_image_mime
            )

        # Record bot reply in memory
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id="bot",
            sender_name=settings.bot_name,
            text=reply_text,
            is_bot=True,
            chat_type="group" if is_group else "user"
        )

        # Send reply back to LINE safely chunked within character limits
        text_messages = [TextMessage(text=c) for c in chunk_line_text(reply_text)]
        try:
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=text_messages
            )
            reply_res = await asyncio.to_thread(messaging_api.reply_message, reply_request)
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
                    push_res = await asyncio.to_thread(messaging_api.push_message, push_req)
                    if push_res and hasattr(push_res, "sent_messages"):
                        for sm in push_res.sent_messages:
                            if hasattr(sm, "id") and sm.id:
                                memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
                    logger.info("Push fallback delivered successfully to [%s]", chat_id)
                except Exception as push_err:
                    logger.error("push_message fallback also failed: %s", push_err)


def parse_image_response(raw_text: str) -> Tuple[str, dict]:
    """
    Parses dual-output from Gemini containing <VISUAL_RECORD> and <REPLY> tags.
    Returns (reply_text, visual_dict) where visual_dict has:
    {"summary": str, "ocr_text": str, "tags": List[str]}
    """
    summary = ""
    ocr_text = ""
    tags: List[str] = []
    reply_text = raw_text

    if "<VISUAL_RECORD>" in raw_text and "</VISUAL_RECORD>" in raw_text:
        record_block = raw_text.split("<VISUAL_RECORD>")[1].split("</VISUAL_RECORD>")[0].strip()
        for line in record_block.splitlines():
            line_str = line.strip()
            if line_str.lower().startswith("summary:"):
                summary = line_str[len("summary:"):].strip()
            elif line_str.lower().startswith("ocr:"):
                ocr_text = line_str[len("ocr:"):].strip()
            elif line_str.lower().startswith("tags:"):
                tags_str = line_str[len("tags:"):].strip()
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    if "<REPLY>" in raw_text and "</REPLY>" in raw_text:
        reply_text = raw_text.split("<REPLY>")[1].split("</REPLY>")[0].strip()
    elif "<VISUAL_RECORD>" in raw_text and "</VISUAL_RECORD>" in raw_text:
        reply_text = raw_text.split("</VISUAL_RECORD>")[1].replace("</REPLY>", "").strip()

    if not summary:
        summary = reply_text[:200].replace("\n", " ").strip()

    return reply_text, {"summary": summary, "ocr_text": ocr_text, "tags": tags}


async def process_image_message(event: MessageEvent):
    """
    Handles an incoming image message event.
    Performs OCR, document/receipt analysis, homework/code solving, or witty Gen Z photo commentary.
    Stores rich visual data in persistent image_vault and sliding history.
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

        # Look up replied message if this image was sent as a quote reply
        quoted_msg = memory_manager.get_message(chat_id, quoted_id) if quoted_id else None
        quoted_image_context = ""
        if quoted_msg:
            q_author = f"{settings.bot_name} (ตัวคุณเอง)" if quoted_msg.is_bot else quoted_msg.sender_name
            quoted_image_context = (
                f"\n[บริบทเพิ่มเติม: ผู้ใช้ส่งรูปภาพนี้มาเพื่อตอบกลับ (Quote Reply) ข้อความของ '{q_author}': \"{quoted_msg.text}\"]\n"
            )

        # Show loading animation (1-on-1 only)
        if settings.enable_loading_animation and not is_group and user_id:
            try:
                anim_req = ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=15)
                await asyncio.to_thread(messaging_api.show_loading_animation, anim_req)
            except Exception as e:
                logger.debug("Loading animation request ignored: %s", e)

        image_mime = detect_image_mime(image_bytes)

        persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
        system_instruction = build_system_prompt(settings.bot_name, persona_key)
        history_context = memory_manager.get_group_memory(chat_id).get_history_formatted(settings.bot_name)
        cross_chat_info = memory_manager.get_cross_chat_context(chat_id, sender_id=user_id)
        if cross_chat_info:
            history_context = f"{history_context}\n\n{cross_chat_info}" if history_context else cross_chat_info

        user_prompt = (
            f"เพื่อนส่งรูปภาพนี้มาในแชท {quoted_image_context}\n"
            "ภารกิจของคุณ: ดูและวิเคราะห์รูปภาพอย่างละเอียดทั้งในแง่ข้อมูล (OCR/ดีเทล) และตอบกลับเพื่อนอย่างเป็นมิตรที่สุด\n\n"
            "ให้ตอบโดยแบ่งออกเป็น 2 ส่วนชัดเจน โดยใช้ Tag ดังนี้:\n\n"
            "<VISUAL_RECORD>\n"
            "summary: [สรุปรายละเอียดของรูปภาพแบบเจาะลึก เช่น สิ่งที่เห็น วัตถุ คน สถานที่ สลิปอะไร เมนูอะไร การบ้านวิชาอะไร โค้ดภาษาอะไร]\n"
            "ocr: [ข้อความทั้งหมดที่อ่านได้จากรูป เช่น ตัวเลขเงิน วันที่เวลา ชื่อบัญชี ข้อความบนป้าย โค้ดในจอ หัวข้อในเอกสาร]\n"
            "tags: [คำสำคัญที่เกี่ยวข้อง คั่นด้วยจุลภาค เช่น สลิป, kbank, 500, แมว, ร้านกาแฟ, โค้ด python]\n"
            "</VISUAL_RECORD>\n\n"
            "<REPLY>\n"
            "[ข้อความตอบกลับเพื่อนในแชท:\n"
            "- สไตล์: เพื่อนสนิท Gen Z แท้ๆ ที่เป็นมิตรและอบอุ่นที่สุด ซัพพอร์ตเพื่อน โดนโน้มน้าวหรือป้ายยาง่าย\n"
            "- กฎ 90/10: 90% พูดคุยสบายๆ เหมือนคนธรรมดา ('เออ', 'อ่อ', 'ว่ะ', 'วะ', 'ดิ', 'ปะ', 'นะ', '55555') และ 10% มุกเฉพาะสถานการณ์\n"
            "- ห้ามยัดเยียดสแลง ห้ามใช้คำว่า bro หรือศัพท์มีมทุกประโยคจน cringe\n"
            "- ห้ามจริตกระเทย/สาวสองเด็ดขาด ห้ามสำนวนแปลนิยาย/หนังฝรั่งเด็ดขาด\n"
            "- สติปัญญา High-IQ: ถ้าเป็นสลิป บิล โค้ด การบ้าน เอกสาร ให้ช่วยคิดเงิน สรุป หรือวิเคราะห์อย่างถูกต้องและแม่นยำ]\n"
            "</REPLY>"
        )

        raw_reply = await asyncio.to_thread(
            gemini_client.generate_chat_response,
            user_message=user_prompt,
            system_instruction=system_instruction,
            history_context=history_context,
            sender_name=sender_name,
            image_bytes=image_bytes,
            mime_type=image_mime
        )

        reply_text, visual_dict = parse_image_response(raw_reply)

        # Save to persistent Visual Memory Vault
        if image_id:
            memory_manager.save_visual_memory(
                image_id=image_id,
                summary=visual_dict["summary"],
                ocr_text=visual_dict["ocr_text"],
                tags=visual_dict["tags"],
                chat_id=chat_id,
                sender_id=user_id,
                sender_name=sender_name
            )

        # Save incoming image action and bot reply into unified memory buffer
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id=user_id or "unknown",
            sender_name=sender_name,
            text=f"[ส่งรูปภาพ] {visual_dict['summary'][:100]}",
            is_bot=False,
            message_id=image_id,
            quoted_message_id=quoted_id,
            image_desc=visual_dict["summary"],
            chat_type="group" if is_group else "user",
        )
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id="bot",
            sender_name=settings.bot_name,
            text=reply_text,
            is_bot=True,
            chat_type="group" if is_group else "user",
        )

        text_messages = [TextMessage(text=c) for c in chunk_line_text(reply_text)]
        try:
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=text_messages
            )
            reply_res = await asyncio.to_thread(messaging_api.reply_message, reply_request)
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
                    push_res = await asyncio.to_thread(messaging_api.push_message, push_req)
                    if push_res and hasattr(push_res, "sent_messages"):
                        for sm in push_res.sent_messages:
                            if hasattr(sm, "id") and sm.id:
                                memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
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
        cross_chat_info = memory_manager.get_cross_chat_context(chat_id, sender_id=user_id)
        if cross_chat_info:
            history_context = f"{history_context}\n\n{cross_chat_info}" if history_context else cross_chat_info

        reply_text = ""

        if category == "pdf":
            user_prompt = (
                f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ PDF ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                f"ช่วยอ่านเนื้อหาในไฟล์นี้ทั้งหมดอย่างละเอียด แล้วสรุปใจความสำคัญ ประเด็นหลัก หรือสิ่งที่น่าสนใจให้ฟัง\n"
                f"- ถ้ามี Action Items, ข้อตกลง, หรือตัวเลขสำคัญให้ดึงมาบอก\n"
                f"- ตอบด้วยภาษาเพื่อนสนิท Gen Z ที่ฉลาด สรุปเนื้อหาเข้าใจง่าย มี bullet points อ่านสบายตา\n"
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
                reply_text = f"เราลองเปิดอ่านไฟล์ Word '{file_name}' แล้วแต่ดูเหมือนไฟล์จะว่างเปล่าหรือเป็นรูปล้วนๆ เลยอ่านข้อความข้างในไม่เจออะ 5555555"
            else:
                user_prompt = (
                    f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ Word (.docx) ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                    f"เนื้อหาในเอกสาร:\n"
                    f"\"\"\"\n{text_content[:40000]}\n\"\"\"\n\n"
                    f"ช่วยอ่านเนื้อหาในเอกสารนี้ แล้วสรุปประเด็นสำคัญ สาระสำคัญ หรืออธิบายสิ่งที่อยู่ในไฟล์ให้เพื่อนฟังแบบเข้าใจง่ายๆ\n"
                    f"สไตล์เพื่อน Gen Z สรุปเนื้อหาคมชัด เข้าใจง่าย ไม่น่าเบื่อ และไม่ใช้ภาษาทางการแข็งทื่อ"
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
                reply_text = f"ไฟล์ Excel '{file_name}' นี้ดูเหมือนไม่มีข้อมูลตัวอักษรหรือตารางที่เราอ่านได้เลย ลองเซฟเป็น CSV หรือส่งเป็นรูปตารางมาดูมั้ยย 5555555"
            else:
                user_prompt = (
                    f"เพื่อนชื่อ '{sender_name}' ส่งไฟล์ตาราง Excel (.xlsx) ชื่อ '{file_name}' มาในแชท {quoted_file_context}\n"
                    f"ข้อมูลตารางในไฟล์:\n"
                    f"\"\"\"\n{sheet_content[:40000]}\n\"\"\"\n\n"
                    f"ช่วยวิเคราะห์และสรุปข้อมูลในตารางนี้ให้เพื่อนฟัง ดึง insight สำคัญ ยอดรวม หรือประเด็นเด่นๆ ออกมาสรุปให้ชัดเจน\n"
                    f"สไตล์เพื่อน Gen Z ฉลาดๆ สรุปประเด็นหลักและตัวเลขสำคัญให้อ่านง่าย ชัดเจน ไม่ตอบทางการน่าเบื่อ"
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
                f"3. ตอบสไตล์เพื่อนซี้ รู้จริง ชัดเจน ตรงประเด็น ห้ามสำนวนนิยาย"
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
                f"ไฟล์ '{file_name}' ({size_kb} KB) อันนี้เรายังแกะเนื้อหาข้างในไม่ได้อะ 5555555\n"
                f"ตอนนี้เรารองรับอ่านไฟล์ PDF, Word (.docx), Excel (.xlsx), CSV, Text/Code (.txt, .json, .py, .md ฯลฯ), รูปภาพ และไฟล์เสียง นะ "
                f"ลองแปลงไฟล์หรือแคปรูปส่งมาให้ดูใหม่อีกทีดิ๊!"
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
            file_desc=f"{file_name} ({file_size} bytes)",
            chat_type="group" if is_group else "user",
        )
        memory_manager.add_message(
            chat_id=chat_id,
            sender_id="bot",
            sender_name=settings.bot_name,
            text=reply_text,
            is_bot=True,
            chat_type="group" if is_group else "user",
        )

        text_messages = [TextMessage(text=c) for c in chunk_line_text(reply_text)]
        try:
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=text_messages
            )
            reply_res = await asyncio.to_thread(messaging_api.reply_message, reply_request)
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
                    push_res = await asyncio.to_thread(messaging_api.push_message, push_req)
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
        f"หวัดดีทุกคนน! โทมัส ({settings.bot_name}) มาละ 5555555 "
        f"เข้ามาร่วมตี้ ชวนคุย ช่วยปั่น ช่วยคิดงานในกลุ่มละนะ ใครมีอะไรสงสัย อยากวางแผน หรืออยากชวนคุยเล่น เรียกเราได้ตลอดเลย\n\n"
        f"📌 วิธีเรียกเรา:\n"
        f"- พิมพ์ชื่อ '{settings.bot_name}' หรือแท็ก @{settings.bot_name}\n"
        f"- ตอบกลับ (Quote Reply) ข้อความเดิมในแชท\n"
        f"- คำสั่งเด็ด: `/news` (สรุปข่าวดังวันนี้), `/plan [เรื่อง]` (วางแผนงานเจาะลึก), `/boost`, `/help`\n\n"
        f"พร้อมคุยแล้วพวกมึง ลุย! 🚀"
    )
    with get_api_client() as api_client:
        messaging_api = MessagingApi(api_client)
        try:
            req = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=welcome_text)]
            )
            reply_res = await asyncio.to_thread(messaging_api.reply_message, req)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
            logger.info("Sent join welcome message to chat [%s]", chat_id)
        except Exception as e:
            logger.error("Failed to send join greeting: %s", e)


async def process_follow_event(event: FollowEvent):
    """Greets a user when they add the bot in a 1-on-1 private chat."""
    chat_id, user_id, _ = extract_chat_and_user_ids(event)
    chat_tracker.register_chat(chat_id, "user")
    welcome_text = (
        f"ว่าไง! 👋 เรา '{settings.bot_name}' เอง 5555555\n"
        f"มีอะไรมาคุย ปรึกษา วางแผนงาน ถามเรื่องเรียน/โค้ด หรือบ่นได้ตลอดนะ ฟีลเพื่อนสนิทคุยกันชิลๆ\n\n"
        f"💡 ลองพิมพ์คุยเล่น หรือลองคำสั่งพวกนี้ดู:\n"
        f"- `/news` : สรุปข่าวดังวันนี้ กระชับ ไม่ตกเทรนด์\n"
        f"- `/plan [เรื่อง]` : วางแผนงานแบบ Step-by-Step ละเอียดชัดเจน\n"
        f"- `/boost [เรื่อง]` : บูสต์พลังใจเวลาหมดไฟ\n"
        f"- `/goal [เรื่อง]` : ตั้งเป้าหมาย SMART Goal ชัดเจน\n"
        f"- `/help` : ดูคำสั่งทั้งหมด\n\n"
        f"หรือดึงเราเข้ากลุ่มไลน์ไปคุยกับแก๊งเพื่อนก็ได้นะ! 🚀"
    )
    with get_api_client() as api_client:
        messaging_api = MessagingApi(api_client)
        try:
            req = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=welcome_text)]
            )
            reply_res = await asyncio.to_thread(messaging_api.reply_message, req)
            if reply_res and hasattr(reply_res, "sent_messages"):
                for sm in reply_res.sent_messages:
                    if hasattr(sm, "id") and sm.id:
                        memory_manager.register_bot_message_id(sm.id, chat_id=chat_id)
        except Exception as e:
            logger.error("Failed to send follow greeting: %s", e)
