"""
Background scheduler for autonomous proactivity:
1. Spontaneous Conversation Starter: Chimes in every 4-5 hours (randomized) during waking hours.
2. Daily Morning News Briefing: Dispatches top Thai news at 08:00 AM (Bangkok time, UTC+7).
"""
import asyncio
import logging
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from linebot.v3.messaging import (
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)

from app.config import settings
from app.chat_tracker import chat_tracker
from app.memory import memory_manager
from app.persona import build_system_prompt
from app.gemini_client import gemini_client
from app.news_service import generate_morning_news_briefing

logger = logging.getLogger("line_gemini_bot")

BANGKOK_TZ = timezone(timedelta(hours=7))


def chunk_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Splits long texts to satisfy LINE's character limits per message."""
    if not text:
        return [""]
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk += ("\n" if current_chunk else "") + paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks[:5]


async def broadcast_push_message(chat_ids: list[str], text: str, reason: str = "scheduler"):
    """Pushes a message safely to a list of registered group chats."""
    if not text or not chat_ids:
        return

    from app.bot import line_config

    text_messages = [TextMessage(text=c) for c in chunk_text(text)]

    with ApiClient(line_config) as api_client:
        messaging_api = MessagingApi(api_client)
        for chat_id in chat_ids:
            try:
                push_req = PushMessageRequest(to=chat_id, messages=text_messages)
                push_res = await asyncio.to_thread(messaging_api.push_message, push_req)

                # Record in memory & register sent message IDs
                memory_manager.add_message(
                    chat_id=chat_id,
                    sender_id="bot",
                    sender_name=settings.bot_name,
                    text=text,
                    is_bot=True
                )
                if push_res and hasattr(push_res, "sent_messages"):
                    for sm in push_res.sent_messages:
                        if hasattr(sm, "id") and sm.id:
                            memory_manager.register_bot_message_id(sm.id)

                logger.info("Broadcasted [%s] to chat [%s]", reason, chat_id)
            except Exception as e:
                logger.error("Failed to push message to [%s]: %s", chat_id, e)


async def send_morning_news_to_all():
    """Generates and broadcasts morning news briefing to all active groups."""
    active_groups = chat_tracker.get_active_group_ids()
    if not active_groups:
        logger.info("Morning news skipped: no registered active groups yet.")
        return

    logger.info("Generating morning news briefing for %d groups...", len(active_groups))
    try:
        briefing_text = await asyncio.to_thread(generate_morning_news_briefing, settings.default_persona)
        await broadcast_push_message(active_groups, briefing_text, reason="morning_news")
    except Exception as e:
        logger.error("Failed to generate or send morning news: %s", e)


async def send_proactive_chatter_to_groups():
    """Sends a witty, spontaneous conversation opener to registered groups."""
    active_groups = chat_tracker.get_active_group_ids()
    if not active_groups:
        return

    logger.info("Running proactive conversation starter for %d groups...", len(active_groups))

    for chat_id in active_groups:
        try:
            mem = memory_manager.get_group_memory(chat_id)
            now = time.time()

            # Prevent chiming in if the bot spoke very recently (< 45 minutes)
            if mem.last_bot_reply_time > 0 and (now - mem.last_bot_reply_time) < 2700:
                logger.debug("Chat [%s] spoke recently, skipping proactive chatter", chat_id)
                continue

            persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
            system_instruction = build_system_prompt(settings.bot_name, persona_key)
            history_context = mem.get_history_formatted(settings.bot_name)

            prompt = f"""คุณคือ {settings.bot_name} เพื่อนสนิทในกลุ่มไลน์ วัยรุ่น Gen Z แท้ๆ
หน้าที่ของคุณ:
เป็นคนทักขึ้นมาเปิดบทสนทนา (Proactive Ice-Breaker) สั้นๆ 1-2 ประโยค เหมือนคนกดพิมพ์ในไลน์หากลุ่มเพื่อน
ตัวอย่างฟีลที่พูดได้:
- แซวความเงียบของกลุ่ม: "พวกแกกกก เงียบกริบขนาดนี้คือทำงานจริงจังหรือแอบหลับเอาดีๆ 5555555", "หายไปไหนกันหมดดด อ่อมมากกกก"
- ชวนคุยเรื่องของกิน: "เที่ยงนี้กินไรกันอะแก แนะนำของกินที สมองไหลละ", "กาแฟแก้วสองต้องเข้าละมั้ย สภาพพพ"
- ชวนคุยเล่นมุกหรือแซวเพื่อน: ถ้าในประวัติการคุยมีชื่อเพื่อน ให้เรียกชื่อหรือแซวเพื่อนคนนั้นได้ตามธรรมชาติ

🚫 ข้อห้ามเด็ดขาด:
- ห้ามใช้สำนวนแปลหนังฝรั่งหรือนิยายเด็ดขาด! (ห้ามพูด: "ไงเพื่อน", "พวกนายเป็นไงบ้าง", "สหาย", "ว่าไงพวก")
- ห้ามแนะนำตัว
- พิมพ์สั้นๆ 1-2 ประโยค ติดสปีดเหมือนแชทวัยรุ่นไทยจริง
"""
            opener = await asyncio.to_thread(
                gemini_client.generate_chat_response,
                user_message=prompt,
                system_instruction=system_instruction,
                history_context=history_context,
                sender_name="ระบบเปิดบทสนทนา"
            )

            await broadcast_push_message([chat_id], opener, reason="proactive_chatter")
            # Stagger messages between multiple groups slightly
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("Error in proactive chatter for chat [%s]: %s", chat_id, e)


class BackgroundScheduler:
    """Manages the periodic loop for morning news and random conversation openers."""
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_news_date: Optional[str] = None
        self._next_chatter_timestamp: float = 0.0

    def _reset_next_chatter_time(self):
        """Randomizes the next chatter interval between min and max hours."""
        interval_seconds = random.uniform(
            settings.chatter_interval_min_hours * 3600,
            settings.chatter_interval_max_hours * 3600
        )
        self._next_chatter_timestamp = time.time() + interval_seconds
        logger.info(
            "Next proactive chatter scheduled in %.1f hours (at ~%s)",
            interval_seconds / 3600,
            datetime.fromtimestamp(self._next_chatter_timestamp, tz=BANGKOK_TZ).strftime("%H:%M:%S")
        )

    async def _loop(self):
        logger.info("Autonomous Background Scheduler started.")
        self._reset_next_chatter_time()

        while self._running:
            try:
                now_bkk = datetime.now(BANGKOK_TZ)
                today_str = now_bkk.strftime("%Y-%m-%d")
                now_ts = time.time()

                # 1. Morning News Check (08:00 AM Bangkok Time)
                if settings.enable_morning_news:
                    if now_bkk.hour >= settings.morning_news_hour and self._last_news_date != today_str:
                        logger.info("Triggering scheduled morning news for %s (Hour: %d)", today_str, now_bkk.hour)
                        self._last_news_date = today_str
                        await send_morning_news_to_all()

                # 2. Random Proactive Chatter Check (Every 4-5 hours during waking hours: 09:00 - 23:00)
                if settings.enable_proactive_chatter:
                    if now_ts >= self._next_chatter_timestamp:
                        # Only initiate during waking hours in Thailand (09:00 to 23:00)
                        if 9 <= now_bkk.hour <= 23:
                            await send_proactive_chatter_to_groups()
                        else:
                            logger.info("Skipping proactive chatter outside waking hours (%02d:%02d)", now_bkk.hour, now_bkk.minute)
                        self._reset_next_chatter_time()

            except Exception as e:
                logger.error("Error in scheduler loop: %s", e)

            # Sleep 60 seconds between checks
            await asyncio.sleep(60)

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                self._task = None
            logger.info("Autonomous Background Scheduler stopped.")


scheduler = BackgroundScheduler()
