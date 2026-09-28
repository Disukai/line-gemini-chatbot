"""
Unit tests for LINE Gemini Chatbot.
Covers command parsing, triggers, memory buffers, persona generation, message chunking, and FastAPI endpoints.
"""
import base64
import hashlib
import hmac
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.config import settings
from app.commands import parse_command, execute_command, HELP_TEXT
from app.persona import (
    build_system_prompt,
    build_plan_prompt,
    build_boost_prompt,
    build_goal_prompt,
    PERSONA_ALIASES
)
from app.memory import MemoryManager, GroupMemory, memory_manager
from app.bot import should_trigger_response, chunk_line_text
from app.main import app


def test_parse_command():
    assert parse_command("/plan สร้างแอปใน 7 วัน") == ("plan", "สร้างแอปใน 7 วัน")
    assert parse_command("/boost") == ("boost", "")
    assert parse_command("/goal วิ่งวันละ 5km") == ("goal", "วิ่งวันละ 5km")
    assert parse_command("/reset") == ("reset", "")
    assert parse_command("/persona chill") == ("persona", "chill")
    assert parse_command("/help") == ("help", "")
    assert parse_command("สวัสดีครับ") is None
    assert parse_command("   /Plan   uppercase test  ") == ("plan", "uppercase test")


def test_persona_prompts():
    prompt = build_system_prompt("จิมมี่", "friend")
    assert "จิมมี่" in prompt
    assert "เพื่อนสนิทในกลุ่มไลน์ที่เป็นคนจริงๆ" in prompt
    assert "ห้ามตอบแบบหุ่นยนต์เด็ดขาด" in prompt

    plan_p = build_plan_prompt("เปิดร้านกาแฟ", has_context=True)
    assert "เปิดร้านกาแฟ" in plan_p
    assert "Action Plan" in plan_p
    assert "Next 15 Minutes Quick-Win" in plan_p
    assert "ประวัติการสนทนา" in plan_p

    boost_p = build_boost_prompt("หมดไฟอ่านหนังสือ", has_context=True)
    assert "หมดไฟอ่านหนังสือ" in boost_p
    assert "Tactical Breakthrough" in boost_p
    assert "บทสนทนาของเพื่อนๆ" in boost_p

    goal_p = build_goal_prompt("เก็บเงิน 1 ล้าน", has_context=False)
    assert "เก็บเงิน 1 ล้าน" in goal_p
    assert "SMART Actionable Goal" in goal_p


def test_chunk_line_text():
    # 1. Short text stays single chunk
    short = "สวัสดีครับเพื่อนๆ"
    assert chunk_line_text(short) == [short]

    # 2. Long text exceeding 4000 characters is split into chunks <= 4000
    long_text = "\n".join([f"ข้อความบรรทัดที่ {i} สำหรับทดสอบระบบ chunking ของ LINE" for i in range(250)])
    assert len(long_text) > 5000
    chunks = chunk_line_text(long_text, max_chunk_size=4000)
    assert len(chunks) > 1
    assert len(chunks) <= 5
    for c in chunks:
        assert len(c) <= 4500


def test_memory_sliding_window():
    mem = GroupMemory(max_history=3)
    mem.add_message("u1", "Harvey", "หวัดดี 1")
    mem.add_message("u2", "Pluem", "หวัดดี 2")
    mem.add_message("bot", "จิมมี่", "หวัดดี 3", is_bot=True)
    mem.add_message("u1", "Harvey", "หวัดดี 4")

    # Should only keep last 3 messages (2, 3, 4)
    assert len(mem.messages) == 3
    assert mem.messages[0].text == "หวัดดี 2"
    assert mem.messages[1].text == "หวัดดี 3"
    assert mem.messages[2].text == "หวัดดี 4"

    history_str = mem.get_history_formatted("จิมมี่")
    assert "[Pluem]: หวัดดี 2" in history_str
    assert "[จิมมี่ (คุณ)]: หวัดดี 3" in history_str
    assert "[Harvey]: หวัดดี 4" in history_str

    mem.clear()
    assert len(mem.messages) == 0


def test_memory_manager_persona_and_profile_cache():
    manager = MemoryManager(max_history=5)
    chat_id = "test_group_1"

    # Default persona
    assert manager.get_persona(chat_id, "friend") == "friend"

    # Override persona
    manager.set_persona(chat_id, "chill")
    assert manager.get_persona(chat_id, "friend") == "chill"

    # Profile cache
    assert manager.get_cached_name("u_test") is None
    manager.cache_name("u_test", "Harvey", is_fallback=False)
    assert manager.get_cached_name("u_test") == "Harvey"

    # Bot message tracking
    assert not manager.is_bot_message("bot_msg_99")
    manager.register_bot_message_id("bot_msg_99")
    assert manager.is_bot_message("bot_msg_99")
    assert manager.has_bot_messages()


def test_should_trigger_response():
    mock_event = MagicMock()
    mock_event.message = MagicMock()
    mock_event.message.quoted_message_id = None
    mock_event.message.mention = None

    bot_name = "จิมมี่"
    nicknames = ["บอท", "จิมมี่", "เจมี่", "gemini"]

    # 1. Private 1-on-1 chat always triggers
    assert should_trigger_response(
        text="สวัสดีวันจันทร์",
        event=mock_event,
        is_group=False,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is True

    # 2. Slash command in group always triggers
    assert should_trigger_response(
        text="/help",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is True

    # 3. Mentioning bot name or nickname in group triggers
    assert should_trigger_response(
        text="จิมมี่ ว่างปะ",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is True

    assert should_trigger_response(
        text="ถาม บอท หน่อย",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is True

    # 4. Normal chatter without bot name should NOT trigger
    assert should_trigger_response(
        text="วันนี้กินข้าวที่ไหนกันดีพวกเรา",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is False

    # 5. Smart trigger mode with question keyword
    assert should_trigger_response(
        text="ไปกินชาบูร้านไหนดีมั้ย",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="smart"
    ) is True

    # 6. Mention of ANOTHER user in group should NOT trigger the bot
    mock_other_mention = MagicMock()
    mock_other_mention.message = MagicMock()
    mock_other_mention.message.quoted_message_id = None
    mock_mentionee_other = MagicMock()
    mock_mentionee_other.is_self = False
    mock_mentionee_other.user_id = "user_other_123"
    mock_mentionee_other.type = "user"
    mock_other_mention.message.mention = MagicMock()
    mock_other_mention.message.mention.mentionees = [mock_mentionee_other]

    assert should_trigger_response(
        text="@Bob พรุ่งนี้เจอกันกี่โมง",
        event=mock_other_mention,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention",
        bot_user_id="bot_user_real"
    ) is False

    # 7. Mention of the bot specifically DOES trigger
    mock_bot_mention = MagicMock()
    mock_bot_mention.message = MagicMock()
    mock_bot_mention.message.quoted_message_id = None
    mock_mentionee_bot = MagicMock()
    mock_mentionee_bot.is_self = True
    mock_mentionee_bot.user_id = "bot_user_real"
    mock_bot_mention.message.mention = MagicMock()
    mock_bot_mention.message.mention.mentionees = [mock_mentionee_bot]

    assert should_trigger_response(
        text="ว่าไง",
        event=mock_bot_mention,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention",
        bot_user_id="bot_user_real"
    ) is True

    # 8. Quoting bot's message in group triggers
    memory_manager.register_bot_message_id("bot_msg_sample_1")
    mock_reply_event = MagicMock()
    mock_reply_event.message = MagicMock()
    mock_reply_event.message.mention = None
    mock_reply_event.message.quoted_message_id = "bot_msg_sample_1"
    assert should_trigger_response(
        text="จริงด้วย",
        event=mock_reply_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is True

    # 9. Quoting another member's message in group does NOT trigger
    mock_member_quote_event = MagicMock()
    mock_member_quote_event.message = MagicMock()
    mock_member_quote_event.message.mention = None
    mock_member_quote_event.message.quoted_message_id = "random_human_msg_id"
    assert should_trigger_response(
        text="โอเค รับทราบครับ",
        event=mock_member_quote_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ) is False


def test_execute_command_static():
    assert "LINE AI เพื่อนซี้ประจำกลุ่ม" in execute_command("help", "", "group1", "Harvey")
    assert "ล้างความจำ" in execute_command("reset", "", "group1", "Harvey")

    # English persona
    assert "ปรับโหมดบุคลิก" in execute_command("persona", "expert", "group1", "Harvey")
    # Thai persona aliases
    assert "ปรับโหมดบุคลิกห้องนี้เป็น [chill]" in execute_command("persona", "ชิล", "group1", "Harvey")
    assert "ปรับโหมดบุคลิกห้องนี้เป็น [expert]" in execute_command("persona", "เซียน", "group1", "Harvey")
    assert "สไตล์ที่มีให้เลือก" in execute_command("persona", "invalid_tone", "group1", "Harvey")

    assert "คำสั่ง `/unknown` ไม่มี" in execute_command("unknown", "", "group1", "Harvey")


def test_fastapi_health_endpoints():
    client = TestClient(app)
    res_root = client.get("/")
    assert res_root.status_code == 200
    data = res_root.json()
    assert data["service"] == "line-gemini-chatbot"
    assert data["model"] == settings.gemini_model

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


def test_webhook_invalid_signature():
    client = TestClient(app)
    settings.line_channel_secret = "dummy_secret"
    res = client.post(
        "/callback",
        content="{}",
        headers={"X-Line-Signature": "invalid_signature"}
    )
    assert res.status_code == 400
    assert "Invalid signature" in res.text


def test_webhook_valid_signature():
    client = TestClient(app)
    secret = "valid_secret_test_123"
    settings.line_channel_secret = secret

    payload = json.dumps({
        "destination": "Ubot123",
        "events": [
            {
                "type": "message",
                "message": {
                    "type": "text",
                    "id": "1001",
                    "text": "หวัดดี"
                },
                "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZR",
                "deliveryContext": {
                    "isRedelivery": False
                },
                "timestamp": 1720000000000,
                "source": {"type": "user", "userId": "Utestuser"},
                "replyToken": "test_reply_token_abc",
                "mode": "active"
            }
        ]
    })

    hash_val = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    signature = base64.b64encode(hash_val).decode("utf-8")

    res = client.post(
        "/callback",
        content=payload,
        headers={"X-Line-Signature": signature}
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_webhook_join_event():
    client = TestClient(app)
    secret = "valid_secret_test_123"
    settings.line_channel_secret = secret

    payload = json.dumps({
        "destination": "Ubot123",
        "events": [
            {
                "type": "join",
                "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZQ",
                "deliveryContext": {
                    "isRedelivery": False
                },
                "timestamp": 1720000000000,
                "source": {"type": "group", "groupId": "Ctestgroup"},
                "replyToken": "test_reply_token_join",
                "mode": "active"
            }
        ]
    })

    hash_val = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    signature = base64.b64encode(hash_val).decode("utf-8")

    res = client.post(
        "/callback",
        content=payload,
        headers={"X-Line-Signature": signature}
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
