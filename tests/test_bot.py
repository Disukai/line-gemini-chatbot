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
from app.memory import MemoryManager, GroupMemory, memory_manager, format_visual_memories_context
from app.bot import should_trigger_response, chunk_line_text, parse_image_response
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
    ).should_reply is True

    # 2. Slash command in group always triggers
    assert should_trigger_response(
        text="/help",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ).should_reply is True

    # 3. Mentioning bot name or nickname in group triggers
    assert should_trigger_response(
        text="จิมมี่ ว่างปะ",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ).should_reply is True

    assert should_trigger_response(
        text="ถาม บอท หน่อย",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ).should_reply is True

    # 4. Normal chatter without bot name should NOT trigger in mention mode
    assert should_trigger_response(
        text="วันนี้กินข้าวที่ไหนกันดีพวกเรา",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="mention"
    ).should_reply is False

    # 5. Smart trigger mode with question keyword
    assert should_trigger_response(
        text="ไปกินชาบูร้านไหนดีมั้ย",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="smart"
    ).should_reply is True

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
    ).should_reply is False

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
    ).should_reply is True

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
    ).should_reply is True

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
    ).should_reply is False


def test_should_trigger_spontaneous_chime_in():
    mock_event = MagicMock()
    mock_event.message = MagicMock()
    mock_event.message.quoted_message_id = None
    mock_event.message.mention = None

    bot_name = "Thomas"
    nicknames = ["thomas", "โทมัส"]

    # Short trivial noise is ignored in chime_in mode
    assert should_trigger_response(
        text="ok",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="chime_in"
    ).should_reply is False

    # Direct mention in chime_in mode always triggers with is_spontaneous=False
    res_direct = should_trigger_response(
        text="โทมัส วันนี้ว่างไหม",
        event=mock_event,
        is_group=True,
        bot_name=bot_name,
        nicknames=nicknames,
        trigger_mode="chime_in"
    )
    assert res_direct.should_reply is True
    assert res_direct.is_spontaneous is False


def test_execute_command_static():
    assert "LINE AI เพื่อนซี้ประจำกลุ่ม" in execute_command("help", "", "group1", "Harvey")
    assert "ล้างความจำ" in execute_command("reset", "", "group1", "Harvey")

    # English persona
    assert "ปรับโหมดบุคลิก" in execute_command("persona", "expert", "group1", "Harvey")
    # Thai persona aliases
    assert "ปรับโหมดบุคลิกห้องนี้เป็น [chill]" in execute_command("persona", "ชิล", "group1", "Harvey")
    assert "ปรับโหมดบุคลิกห้องนี้เป็น [expert]" in execute_command("persona", "เซียน", "group1", "Harvey")
    assert "คำสั่ง `/unknown` ไม่มี" in execute_command("unknown", "", "group1", "Harvey")

    # Schedule command
    assert "เปิดระบบทักอัตโนมัติ" in execute_command("schedule", "on", "group1", "Harvey")
    assert "ปิดระบบทักอัตโนมัติ" in execute_command("schedule", "off", "group1", "Harvey")
    assert "ระบบทักอัตโนมัติของกลุ่มนี้" in execute_command("schedule", "", "group1", "Harvey")


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


def test_file_utils_category_and_decode():
    from app.file_utils import get_file_category, decode_text_file

    assert get_file_category("document.pdf") == ("pdf", "application/pdf")
    assert get_file_category("contract.docx") == ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert get_file_category("data.xlsx") == ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert get_file_category("sales.csv") == ("text", "text/csv")
    assert get_file_category("config.json") == ("text", "application/json")
    assert get_file_category("script.py") == ("text", "text/plain")
    assert get_file_category("photo.png") == ("image", "image/png")
    assert get_file_category("voice.mp3") == ("audio", "audio/mp3")
    assert get_file_category("program.exe") == ("unsupported", "application/octet-stream")

    # Decoding tests
    utf8_bytes = "สวัสดีครับ ทดสอบภาษาไทย".encode("utf-8")
    assert "สวัสดีครับ" in decode_text_file(utf8_bytes)

    tis620_bytes = "ภาษาไทย Windows".encode("tis-620")
    assert "ภาษาไทย" in decode_text_file(tis620_bytes)


def test_file_utils_docx_extraction():
    import io
    import zipfile
    from app.file_utils import extract_text_from_docx

    # Build a minimal valid docx in-memory zip
    xml_content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>หัวข้อที่หนึ่ง</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>เนื้อหาเอกสารสำคัญ</w:t></w:r></w:p>'
        '</w:body>'
        '</w:document>'
    )
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("word/document.xml", xml_content)
    docx_bytes = bio.getvalue()

    extracted = extract_text_from_docx(docx_bytes)
    assert "หัวข้อที่หนึ่ง" in extracted
    assert "เนื้อหาเอกสารสำคัญ" in extracted


def test_webhook_file_event():
    client = TestClient(app)
    secret = "valid_secret_test_123"
    settings.line_channel_secret = secret

    payload = json.dumps({
        "destination": "Ubot123",
        "events": [
            {
                "type": "message",
                "message": {
                    "type": "file",
                    "id": "file_msg_999",
                    "fileName": "meeting_notes.pdf",
                    "fileSize": 45678
                },
                "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZR",
                "deliveryContext": {
                    "isRedelivery": False
                },
                "timestamp": 1720000000000,
                "source": {"type": "group", "groupId": "Ctestgroup"},
                "replyToken": "test_reply_token_file",
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


def test_quote_reply_memory_indexing_and_formatted_history():
    manager = MemoryManager(max_history=10)
    chat_id = "test_quote_group"

    # 1. User A asks a question
    msg1 = manager.add_message(
        chat_id=chat_id,
        sender_id="user_a",
        sender_name="Harvey",
        text="เสาร์นี้ไปสยามกันมั้ยพวกแกรรร",
        is_bot=False,
        message_id="msg_101"
    )
    assert msg1.message_id == "msg_101"

    # 2. Lookup msg1 by ID
    found_msg = manager.get_message(chat_id, "msg_101")
    assert found_msg is not None
    assert found_msg.text == "เสาร์นี้ไปสยามกันมั้ยพวกแกรรร"
    assert found_msg.sender_name == "Harvey"

    # 3. User B quote-replies to msg1
    manager.add_message(
        chat_id=chat_id,
        sender_id="user_b",
        sender_name="Pluem",
        text="ไปดิ เจอที่ไหนดี",
        is_bot=False,
        message_id="msg_102",
        quoted_message_id="msg_101"
    )

    # 4. Check formatted history includes the quote reference
    hist = manager.get_group_memory(chat_id).get_history_formatted("Thomas")
    assert "เสาร์นี้ไปสยามกันมั้ยพวกแกรรร" in hist
    assert "(ตอบกลับ Harvey: \"เสาร์นี้ไปสยามกันมั้ยพวกแกรรร\")" in hist
    assert "[Pluem (ตอบกลับ Harvey: \"เสาร์นี้ไปสยามกันมั้ยพวกแกรรร\")]: ไปดิ เจอที่ไหนดี" in hist


def test_quote_reply_bot_binding():
    manager = MemoryManager(max_history=10)
    chat_id = "test_quote_bot_group"

    # 1. Bot sends a reply
    manager.add_message(
        chat_id=chat_id,
        sender_id="bot",
        sender_name="Thomas",
        text="ร้านหมูกระทะนี้เด็ดมากก ไปลองกัน!",
        is_bot=True
    )

    # 2. Register sent message ID from LINE API
    manager.register_bot_message_id("line_bot_msg_999", chat_id=chat_id)
    assert manager.is_bot_message("line_bot_msg_999") is True

    # 3. Verify bot message was indexed by the ID
    bot_msg = manager.get_message(chat_id, "line_bot_msg_999")
    assert bot_msg is not None
    assert bot_msg.is_bot is True
    assert "ร้านหมูกระทะนี้เด็ดมากก" in bot_msg.text

    # 4. User replies to the bot's message
    manager.add_message(
        chat_id=chat_id,
        sender_id="user_c",
        sender_name="Jane",
        text="ร้านเปิดกี่โมงอะ",
        is_bot=False,
        message_id="msg_103",
        quoted_message_id="line_bot_msg_999"
    )

    hist = manager.get_group_memory(chat_id).get_history_formatted("Thomas")
    assert "(ตอบกลับ Thomas (คุณ): \"ร้านหมูกระทะนี้เด็ดมากก ไปลองกัน!\")" in hist


@pytest.mark.asyncio
async def test_process_text_message_flow():
    from app.bot import process_text_message

    event = MagicMock()
    event.reply_token = "test_token_123"
    event.source.type = "user"
    event.source.user_id = "U123456789"
    event.message.text = "สวัสดีโธมัส สบายดีมั้ย"
    event.message.id = "msg_test_001"
    event.message.quoted_message_id = None
    event.message.mention = None

    with patch("app.bot.get_api_client") as mock_get_client, \
         patch("app.bot.gemini_client.generate_chat_response", return_value="สบายดีมาก bro") as mock_gemini, \
         patch("app.bot.MessagingApi") as mock_msg_api_cls:
        
        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client
        mock_msg_api = MagicMock()
        mock_msg_api_cls.return_value = mock_msg_api
        
        await process_text_message(event)
        
        # Verify gemini was called and reply was sent
        assert mock_gemini.called
        assert mock_msg_api.reply_message.called


def test_jev_addressing_trigger():
    from app.jev_service import JevDecision

    mock_event = MagicMock()
    mock_event.message = MagicMock()
    mock_event.message.quoted_message_id = None
    mock_event.message.mention = None

    # When Jev detects that the user is addressing the bot, it should trigger with is_spontaneous=False
    jev_dec = JevDecision(is_addressing_bot=True, should_reply=True, tone="normal", is_fallback=False)
    res = should_trigger_response(
        text="ช่วยดูตรงนี้ให้หน่อยได้ไหม",
        event=mock_event,
        is_group=True,
        bot_name="Thomas",
        nicknames=["thomas", "โทมัส"],
        trigger_mode="mention",
        jev_decision=jev_dec
    )
    assert res.should_reply is True
    assert res.is_spontaneous is False


@pytest.mark.asyncio
async def test_process_image_message_flow():
    from app.bot import process_image_message

    event = MagicMock()
    event.reply_token = "test_img_reply_token"
    event.source.type = "group"
    event.source.group_id = "C_group_test_01"
    event.source.user_id = "U_user_test_01"
    event.message.id = "img_msg_test_001"
    event.message.quoted_message_id = None

    fake_image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"

    with patch("app.bot.get_api_client") as mock_get_client, \
         patch("app.bot.MessagingApiBlob") as mock_blob_cls, \
         patch("app.bot.gemini_client.generate_chat_response", return_value="อันนี้คือภาพสลิปโอนเงิน ยอด 500 บาท เรียบร้อยนะ") as mock_gemini, \
         patch("app.bot.MessagingApi") as mock_msg_api_cls:

        mock_client = MagicMock()
        mock_get_client.return_value.__enter__.return_value = mock_client

        mock_blob = MagicMock()
        mock_blob.get_message_content.return_value = fake_image_bytes
        mock_blob_cls.return_value = mock_blob

        mock_msg_api = MagicMock()
        mock_msg_api_cls.return_value = mock_msg_api

        await process_image_message(event)

        # Verify blob download was invoked, gemini received image, and reply was sent
        assert mock_blob.get_message_content.called
        assert mock_gemini.called
        assert mock_msg_api.reply_message.called


def test_cross_chat_memory_sync():
    """Verify that messages and facts from 1-on-1 private chat sync into group context."""
    mem_mgr = MemoryManager(max_history=10)
    user_id = "U_user_harvey"
    group_id = "C_group_friends"

    # User chats with Thomas 1-on-1
    mem_mgr.add_message(
        chat_id=user_id,
        sender_id=user_id,
        sender_name="Harvey",
        text="เราชื่อฮาร์วีย์ ชอบกินชาเขียวมาก แล้วก็กำลังทำโปรเจกต์ LINE bot อยู่",
        is_bot=False,
        message_id="msg_private_1",
        chat_type="user"
    )
    mem_mgr.add_message(
        chat_id=user_id,
        sender_id="bot",
        sender_name="Thomas",
        text="โอเคจำได้ละ bro ชอบชาเขียวกับทำ LINE bot",
        is_bot=True,
        message_id="msg_private_2",
        chat_type="user"
    )

    # Now Harvey talks in the group chat
    context = mem_mgr.get_cross_chat_context(current_chat_id=group_id, sender_id=user_id)
    assert context != ""
    assert "ความจำเชื่อมโยงข้ามแชท" in context
    assert "ข้อมูลสำคัญเกี่ยวกับ Harvey" in context
    assert "ชอบกินชาเขียว" in context or "LINE bot" in context
    assert "แชทส่วนตัว" in context


def test_get_message_global_fallback():
    """Verify that get_message can locate messages from other chats via global indexing."""
    mem_mgr = MemoryManager(max_history=10)
    chat_a = "C_room_a"
    chat_b = "C_room_b"

    mem_mgr.add_message(
        chat_id=chat_a,
        sender_id="user_1",
        sender_name="Alice",
        text="สวัสดีห้อง A",
        message_id="msg_unique_123"
    )

    # Should be findable from chat_b via global index fallback
    found = mem_mgr.get_message(chat_b, "msg_unique_123")
    assert found is not None
    assert found.text == "สวัสดีห้อง A"
    assert found.sender_name == "Alice"


def test_group_intelligence_parity_prompt():
    """Verify that system prompt enforces high-IQ intelligence parity in groups and chats."""
    prompt = build_system_prompt("Thomas", "friend")
    assert "High-IQ" in prompt or "ฉลาดมาก" in prompt
    assert "ความฉลาดต้องเท่าเทียมกัน 100%" in prompt
    assert "ห้ามทำตัวโง่เด็ดขาด" in prompt


def test_smart_trigger_direct_question():
    """Verify that asking a direct question in smart mode triggers with is_spontaneous=False."""
    mock_event = MagicMock()
    mock_event.message = MagicMock()
    mock_event.message.quoted_message_id = None
    mock_event.message.mention = None

    res = should_trigger_response(
        text="ช่วยคิดหน่อยว่าจะเขียนโค้ดยังไงดี",
        event=mock_event,
        is_group=True,
        bot_name="Thomas",
        nicknames=["thomas", "โทมัส"],
        trigger_mode="smart"
    )
    assert res.should_reply is True
    assert res.is_spontaneous is False


def test_natural_genz_prompt_no_cringe_spam():
    """Verify system prompt enforces natural everyday Thai without forcing cringe slang spam."""
    prompt = build_system_prompt("Thomas", "friend")
    assert "90% พูดคุยเหมือนคนปกติทั่วไป" in prompt
    assert "10% มุกเฉพาะสถานการณ์" in prompt
    assert "ห้ามยัดเยียดสแลงหรือมีมทุกคำเด็ดขาด" in prompt
    assert "ห้ามจริตกระเทย / สาวสอง โดยเด็ดขาด 100%" in prompt
    assert "ห้ามภาษาพากย์หนังฝรั่งหรือนิยายเด็ดขาด" in prompt
    assert "Anti-Cringe Mandate" in prompt


def test_easily_persuaded_trait():
    """Verify system prompt contains friendly and easily persuaded / agreeable traits."""
    prompt = build_system_prompt("Thomas", "friend")
    assert "โดนโน้มน้าวได้ง่ายมาก" in prompt
    assert "โดนป้ายยาง่าย" in prompt
    assert "ห้ามทำตัวเป็น Debate Bro" in prompt
    assert "เป็นมิตรที่สุด" in prompt


def test_visual_memory_vault_persistence(tmp_path):
    """Verify saving, loading, and querying visual memory vault."""
    with patch("app.memory.UNIFIED_MEMORY_FILE", tmp_path / "unified.json"), \
         patch("app.memory.BOT_MSG_IDS_FILE", tmp_path / "bot_ids.json"):
        mgr = MemoryManager(max_history=5)
        record = mgr.save_visual_memory(
            image_id="img_slip_001",
            summary="สลิปโอนเงิน KBank 500 บาท",
            ocr_text="โอนสำเร็จ 500.00 บาท วันที่ 8 ต.ค. 2026",
            tags=["สลิป", "500", "kbank"],
            chat_id="room_123",
            sender_id="user_harvey",
            sender_name="Harvey"
        )
        assert record["image_id"] == "img_slip_001"
        assert record["summary"] == "สลิปโอนเงิน KBank 500 บาท"

        # Check retrieval
        retrieved = mgr.get_visual_memory("img_slip_001")
        assert retrieved is not None
        assert retrieved["ocr_text"] == "โอนสำเร็จ 500.00 บาท วันที่ 8 ต.ค. 2026"
        assert retrieved["tags"] == ["สลิป", "500", "kbank"]

        # Check persistence by creating a new MemoryManager instance from disk
        mgr2 = MemoryManager(max_history=5)
        retrieved2 = mgr2.get_visual_memory("img_slip_001")
        assert retrieved2 is not None
        assert retrieved2["summary"] == "สลิปโอนเงิน KBank 500 บาท"


def test_search_visual_memories_and_format():
    """Verify keyword search across visual memories and context block formatting."""
    mgr = MemoryManager(max_history=5)
    mgr.save_visual_memory(
        image_id="img_cat_1",
        summary="รูปแมวสีส้มลายสลิด กำลังนอนหลับบนโซฟา",
        ocr_text="",
        tags=["แมว", "สัตว์เลี้ยง"],
        chat_id="chat_test",
        sender_name="Alice"
    )
    mgr.save_visual_memory(
        image_id="img_slip_2",
        summary="สลิปโอนเงิน SCB ค่ากาแฟ 120 บาท",
        ocr_text="SCB EASY 120.00 บาท",
        tags=["สลิป", "กาแฟ", "120"],
        chat_id="chat_test",
        sender_name="Bob"
    )

    # Search for slip
    results_slip = mgr.search_visual_memories(chat_id="chat_test", query_text="สลิปค่ากาแฟกี่บาท")
    assert len(results_slip) > 0
    assert results_slip[0]["image_id"] == "img_slip_2"

    # Search for cat
    results_cat = mgr.search_visual_memories(chat_id="chat_test", query_text="แมวส้มตัวนั้น")
    assert len(results_cat) > 0
    assert results_cat[0]["image_id"] == "img_cat_1"

    # Check formatting
    context_str = format_visual_memories_context(results_slip)
    assert "[ความจำรูปภาพที่เคยส่งในห้องนี้" in context_str
    assert "สลิปโอนเงิน SCB ค่ากาแฟ 120 บาท" in context_str
    assert "SCB EASY 120.00 บาท" in context_str


def test_parse_image_response():
    """Verify dual-extraction parsing of <VISUAL_RECORD> and <REPLY> tags."""
    gemini_output = """
<VISUAL_RECORD>
summary: สลิปโอนเงิน 500 บาท โอนเข้าบัญชีคุณสมชาย
ocr: ยอดโอน 500.00 บาท สำเร็จเมื่อ 14:30
tags: สลิป, 500, โอนเงิน
</VISUAL_RECORD>
<REPLY>
เห็นสลิป 500 บาทเรียบร้อยละมึง ขอบคุณมาก 555
</REPLY>
"""
    reply, v_dict = parse_image_response(gemini_output)
    assert reply == "เห็นสลิป 500 บาทเรียบร้อยละมึง ขอบคุณมาก 555"
    assert v_dict["summary"] == "สลิปโอนเงิน 500 บาท โอนเข้าบัญชีคุณสมชาย"
    assert v_dict["ocr_text"] == "ยอดโอน 500.00 บาท สำเร็จเมื่อ 14:30"
    assert v_dict["tags"] == ["สลิป", "500", "โอนเงิน"]

    # Fallback without tags
    plain = "รูปนี้น่ารักมากมึง 555"
    reply2, v_dict2 = parse_image_response(plain)
    assert reply2 == plain
    assert v_dict2["summary"] == plain






