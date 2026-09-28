"""
Slash command dispatcher for LINE Gemini Chatbot.
Handles /plan, /boost, /goal, /reset, /persona, and /help.
"""
import logging
from typing import Optional, Tuple
from app.config import settings
from app.persona import (
    build_system_prompt,
    build_plan_prompt,
    build_boost_prompt,
    build_goal_prompt,
    PERSONA_STYLES,
    PERSONA_ALIASES
)
from app.memory import memory_manager
from app.gemini_client import gemini_client

logger = logging.getLogger("line_gemini_bot")


HELP_TEXT = f"""✨ **LINE AI เพื่อนซี้ประจำกลุ่ม (Gemini 3.8 Flash)** ✨
ผมชื่อ "{settings.bot_name}" สมาชิกกลุ่มสายซัพพอร์ต คุยเล่นได้ ปรึกษาได้ แซวได้เหมือนคนจริงๆ 555

💡 **วิธีเรียกคุยในกลุ่ม:**
1. แท็กชื่อ @{settings.bot_name} หรือพิมพ์ชื่อ "{settings.bot_name}" / "บอท" ในข้อความ
2. ตอบกลับ (Reply/Quote) ข้อความของผม
3. ร่วมแจมคุยหรือแซวเป็นระยะตามธรรมชาติ (โดยไม่ต้องแท็ก) ในโหมดแจมคุย
4. คุยในแชทส่วนตัวแบบ 1 ต่อ 1 ได้ตลอดเวลา

🛠️ **คำสั่งพิเศษ (/Commands):**
- `/news` : อัปเดตและสรุปข่าวดังวันนี้ล่าสุด สไตล์เพื่อนเล่าให้ฟังทันที
- `/schedule [on|off]` : เปิด/ปิด ระบบสุ่มทักเปิดบทสนทนาทุก 4-5 ชม. และสรุปข่าวดังยามเช้า
- `/plan [เรื่องที่ต้องการวางแผน]` : วางแผนกลยุทธ์ Action Plan ละเอียดจัดเต็ม ขั้นตอน และจุดตาย
- `/boost [เรื่อง/ปัญหา/ความเหนื่อย]` : บูสต์พลังใจ ปลุกไฟ พร้อมทริคทะลวงจุดตัน ลุยต่อทันที
- `/goal [เป้าหมายที่อยากทำ]` : แปลงเป้าหมายเป็น SMART Goal + เช็คลิสต์ 3 สิ่งที่ต้องทำวันนี้
- `/mode [chime_in|mention|all]` : ปรับโหมดการตอบในกลุ่ม (แจมคุยเป็นธรรมชาติ / ตอบเมื่อแท็กเท่านั้น / ตอบทุกข้อความ)
- `/persona [friend|chill|expert|snarky]` : ปรับบุคลิกของผม (เพื่อนซี้ / สายชิล / มือโปร / สายกวน)
- `/reset` : ล้างความจำการคุยในกลุ่มนี้ (เริ่มคุยหัวข้อใหม่)
- `/help` : ดูคู่มือคำสั่งนี้อีกรอบ

จัดมาได้เลยเพื่อน พร้อมลุย! 🚀
"""


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    """Checks if message is a slash command and extracts (command, args)."""
    clean_text = text.strip()
    if not clean_text.startswith("/"):
        return None

    parts = clean_text.split(maxsplit=1)
    cmd = parts[0][1:].lower()  # strip the leading '/'
    args = parts[1].strip() if len(parts) > 1 else ""
    return (cmd, args)


def execute_command(
    cmd: str,
    args: str,
    chat_id: str,
    sender_name: str,
    history_context: str = ""
) -> str:
    """Executes the specific slash command and returns response text."""
    if cmd == "help":
        return HELP_TEXT

    elif cmd == "reset":
        memory_manager.clear_memory(chat_id)
        return f"🧹 เรียบร้อยคุณ {sender_name}! ล้างความจำบริบทแชทของห้องนี้ให้หมดแล้ว เริ่มคุยหัวข้อใหม่ได้เลย 555"

    elif cmd == "mode":
        target = args.strip().lower()
        mode_descriptions = {
            "chime_in": "ร่วมแจมคุยเป็นธรรมชาติ (ตอบเมื่อแท็ก + สุ่มแจมคุย/แซว/ตอบคำถามโดยไม่ต้องแท็ก)",
            "mention": "ตอบเฉพาะเมื่อถูกแท็กหรือเรียกชื่อเท่านั้น",
            "all": "ตอบทุกข้อความในกลุ่ม",
        }
        mode_aliases = {
            "chime_in": "chime_in",
            "แจม": "chime_in",
            "คุย": "chime_in",
            "ธรรมชาติ": "chime_in",
            "auto": "chime_in",
            "smart": "chime_in",
            "mention": "mention",
            "แท็ก": "mention",
            "เรียก": "mention",
            "all": "all",
            "ทุกข้อความ": "all",
            "ทั้งหมด": "all",
        }
        canonical = mode_aliases.get(target)
        if canonical:
            memory_manager.set_trigger_mode(chat_id, canonical)
            return f"⚙️ ปรับโหมดการตอบของห้องนี้เป็น [{canonical}]:\n👉 {mode_descriptions[canonical]} เรียบร้อยแล้วเพื่อน!"
        else:
            current = memory_manager.get_trigger_mode(chat_id, settings.group_trigger_mode)
            return (
                f"โหมดการตอบในกลุ่มที่มีให้เลือก:\n"
                f"- `/mode chime_in` (หรือ `/mode แจม`) : ตอบเมื่อแท็ก + ร่วมแจมคุยบางทีอย่างเป็นธรรมชาติ (ค่าเริ่มต้น แนะนำ!)\n"
                f"- `/mode mention` (หรือ `/mode แท็ก`) : ตอบเฉพาะเมื่อถูกแท็กหรือเรียกชื่อเท่านั้น\n"
                f"- `/mode all` : ตอบทุกข้อความในกลุ่ม\n\n"
                f"📌 โหมดปัจจุบันของห้องนี้: [{current}]"
            )

    elif cmd == "persona":
        target = args.strip().lower()
        canonical_persona = PERSONA_ALIASES.get(target)
        if canonical_persona:
            memory_manager.set_persona(chat_id, canonical_persona)
            return f"🎭 จัดไป! ปรับโหมดบุคลิกห้องนี้เป็น [{canonical_persona}] เรียบร้อย ลองทักมาคุยดูสิเพื่อน"
        else:
            options = ", ".join(PERSONA_STYLES.keys())
            return (
                f"สไตล์ที่มีให้เลือกมีดังนี้เพื่อน: {options}\n"
                f"หรือภาษาไทย: เพื่อน (friend), ชิล (chill), เซียน/โปร (expert), กวน (snarky)\n"
                f"เช่น พิมพ์ `/persona ชิล` หรือ `/persona expert`"
            )

    elif cmd == "plan":
        if not args:
            return (
                "จะให้ช่วยวางแผนเรื่องอะไรดีเพื่อน? รบกวนพิมพ์บอกหน่อย เช่น "
                "`/plan เตรียมอ่านสอบ ก.พ. ใน 30 วัน` หรือ `/plan เปิดร้านกาแฟเล็กๆ ใช้งบ 1 แสน` "
                "(หรือถ้าจะให้วางแผนเรื่องที่เพิ่งคุยกันในกลุ่ม พิมพ์ `/plan เรื่องที่คุยกันเมื่อกี้` มาได้เลยนะ!)"
            )
        has_history = bool(history_context.strip())
        system_inst = build_plan_prompt(args, has_context=has_history)
        return gemini_client.generate_chat_response(
            user_message=f"ช่วยวางแผนเรื่องนี้ให้หน่อย: {args}",
            system_instruction=system_inst,
            history_context=history_context,
            sender_name=sender_name
        )

    elif cmd == "boost":
        situation = args if args else "เหนื่อย หมดไฟ อยากได้พลังใจลุยงานต่อ"
        has_history = bool(history_context.strip())
        system_inst = build_boost_prompt(situation, has_context=has_history)
        return gemini_client.generate_chat_response(
            user_message=f"ขอกำลังใจและแนวทางบูสต์พลังเรื่องนี้หน่อย: {situation}",
            system_instruction=system_inst,
            history_context=history_context,
            sender_name=sender_name
        )

    elif cmd == "goal":
        if not args:
            return (
                "เป้าหมายของคุณคืออะไรเพื่อน? พิมพ์บอกมาหน่อย เช่น "
                "`/goal เก็บเงินให้ได้ 1 แสนในปีนี้` หรือ `/goal ลดน้ำหนัก 5 กิโลใน 2 เดือน`"
            )
        has_history = bool(history_context.strip())
        system_inst = build_goal_prompt(args, has_context=has_history)
        return gemini_client.generate_chat_response(
            user_message=f"ช่วยแปลงเป้าหมายนี้ให้เป็นรูปธรรมหน่อย: {args}",
            system_instruction=system_inst,
            history_context=history_context,
            sender_name=sender_name
        )

    elif cmd in ("news", "ข่าว", "อัปเดตข่าว"):
        from app.news_service import generate_morning_news_briefing
        persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
        return generate_morning_news_briefing(persona_key)

    elif cmd in ("chatter", "ทัก", "เปิดประเด็น"):
        persona_key = memory_manager.get_persona(chat_id, settings.default_persona)
        system_inst = build_system_prompt(settings.bot_name, persona_key)
        prompt = (
            f"คุณคือ {settings.bot_name} เพื่อนสนิทในกลุ่มไลน์\n"
            f"ช่วยทักขึ้นมาเปิดประเด็นคุยกับเพื่อนในกลุ่มหน่อย 1-2 ประโยค สไตล์เพื่อนสนิทกวนๆ ฮาๆ คุยสนุก ชวนคุยหรือถามอะไรก็ได้"
        )
        return gemini_client.generate_chat_response(
            user_message=prompt,
            system_instruction=system_inst,
            history_context=history_context,
            sender_name="ระบบเปิดบทสนทนา"
        )

    elif cmd == "schedule":
        from app.chat_tracker import chat_tracker
        target = args.strip().lower()
        if target in ("on", "เปิด", "true", "1"):
            chat_tracker.set_schedule_enabled(chat_id, True)
            return "🔔 เปิดระบบทักอัตโนมัติ (ทักเปิดบทสนทนาทุก 4-5 ชม. + สรุปข่าวดังทุกเช้า) สำหรับกลุ่มนี้แล้วครับ!"
        elif target in ("off", "ปิด", "false", "0"):
            chat_tracker.set_schedule_enabled(chat_id, False)
            return "🔕 ปิดระบบทักอัตโนมัติสำหรับกลุ่มนี้แล้วครับ (บอทจะตอบเฉพาะเวลาคุยในกลุ่มตามปกติ)"
        else:
            status = "เปิดใช้งานอยู่ 🔔" if chat_tracker.is_schedule_enabled(chat_id) else "ปิดอยู่ 🔕"
            return (
                f"⏰ **ระบบทักอัตโนมัติของกลุ่มนี้:** {status}\n\n"
                f"- สรุปข่าวดังยามเช้า: ทุกวันเวลา ~08:00 น.\n"
                f"- สุ่มทักเปิดบทสนทนา: ทุกๆ 4-5 ชั่วโมง (ช่วงเวลา 09:00 - 23:00 น.)\n\n"
                f"💡 วิธีเปิด/ปิด:\n"
                f"- พิมพ์ `/schedule on` : เปิดระบบ\n"
                f"- พิมพ์ `/schedule off` : ปิดระบบ"
            )

    return f"คำสั่ง `/{cmd}` ไม่มีนะเพื่อน ลองพิมพ์ `/help` เพื่อดูคำสั่งทั้งหมดได้เลย"
