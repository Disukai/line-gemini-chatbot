"""
News service module for fetching and summarizing top daily news in Thailand.
Uses Google News RSS Thailand with Gemini to deliver witty, friendly morning briefings.
"""
import logging
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

from app.config import settings
from app.persona import build_system_prompt
from app.gemini_client import gemini_client

logger = logging.getLogger("line_gemini_bot")

GOOGLE_NEWS_THAI_RSS = "https://news.google.com/rss?hl=th&gl=TH&ceid=TH:th"


def fetch_top_thai_news(max_items: int = 6) -> List[Dict[str, str]]:
    """
    Fetches the latest trending news headlines from Google News RSS Thailand.
    Returns a list of dicts with 'title', 'link', 'source', and 'pub_date'.
    """
    news_items = []
    try:
        req = urllib.request.Request(
            GOOGLE_NEWS_THAI_RSS,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LineGeminiBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")

        for item in items[:max_items]:
            raw_title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""

            # Extract source if available (e.g. "พาดหัวข่าว - สำนักข่าว")
            source = ""
            if " - " in raw_title:
                parts = raw_title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()
            else:
                title = raw_title.strip()

            if title:
                news_items.append({
                    "title": title,
                    "source": source,
                    "link": link,
                    "pub_date": pub_date
                })

        logger.info("Successfully fetched %d news items from Google News RSS", len(news_items))
    except Exception as e:
        logger.error("Error fetching Google News RSS: %s", e)

    return news_items


def generate_morning_news_briefing(persona_key: str = "friend") -> str:
    """
    Fetches the latest news and uses Gemini to format a punchy, engaging
    morning news briefing in the persona of Thomas (friendly, witty, Thai banter).
    """
    items = fetch_top_thai_news(max_items=7)

    if not items:
        # Fallback if news RSS is temporarily unreachable
        headlines_summary = "- สถานการณ์บ้านเมืองและเศรษฐกิจวันนี้มีหลายประเด็นน่าติดตาม\n- สภาพอากาศและฝนฟ้าคะนองในหลายพื้นที่"
    else:
        headlines_summary = "\n".join([f"- {it['title']} ({it['source']})" if it['source'] else f"- {it['title']}" for it in items])

    prompt = f"""นี่คือประเด็นข่าวเด่นล่าสุดของวันนี้:
{headlines_summary}

ภารกิจของคุณในฐานะ {settings.bot_name} (เพื่อนสนิทสาย Gen Z ในกลุ่มไลน์):
1. ทักทายเพื่อนๆ ยามเช้าแบบคนในโซเชียลคุยกัน (เช่น "มอนิ่งงงพวกแกรรร ตื่นมาเสิร์ฟข่าวฉ่ำๆ ละ 5555555", "ฮัลโหลลล วันนี้มีเรื่องเม้าท์ฉ่ำมากกก")
2. หยิบประเด็นข่าวดัง 3 เรื่องที่พีคที่สุดของวันนี้มาเม้าท์ให้เพื่อนฟัง สั้น กระชับ ฟีลสรุปข่าวไวแบบ TikToker / X
3. ใช้ศัพท์วัยรุ่น Gen Z / Gen Alpha ไทย (เช่น ทำถึง, นอยด์อะ, ช็อตฟีล, ฉ่ำ, ตัวแม่, อ่อม, ของแทร่) เม้าท์หรือแซวประกอบข่าวละ 1 บรรทัด
4. ตบท้ายด้วยการชวนเพื่อนคุยสั้นๆ

🚫 ข้อห้ามเด็ดขาด:
- ห้ามพูดเป็นทางการเด็ดขาด! ห้ามทำตัวเป็นผู้ประกาศข่าวทีวี
- ห้ามใช้สำนวนแปลหนังฝรั่ง เช่น "สวัสดีตอนเช้าเพื่อนๆ", "นี่คือรายงานข่าว", "ไปลุยกันเถอะ"
- ให้อ่านง่าย เว้นบรรทัดสวยงาม ไม่ยาวเยิ่นเย้อ
"""

    system_instruction = build_system_prompt(settings.bot_name, persona_key)

    briefing = gemini_client.generate_chat_response(
        user_message=prompt,
        system_instruction=system_instruction,
        sender_name="ระบบอัปเดตยามเช้า"
    )

    return briefing
