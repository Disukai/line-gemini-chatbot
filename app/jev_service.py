"""
TypeSafe Jev (System One AI) Service module.
Provides fast (< 0.5s), calibrated semantic decisions for:
1. Intent & Addressing detection: whether a message is addressing or calling the bot.
2. Group chime-in decision: whether the bot should naturally speak up.
3. Tone classification: dynamically routing to 'normal' (ordinary human) vs 'banter' (shitpost/brainrot).
"""
import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List

from app.config import settings

logger = logging.getLogger("line_gemini_bot")

try:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul
    TYPESAFE_AVAILABLE = True
except ImportError:
    TYPESAFE_AVAILABLE = False
    logger.warning("typesafe-sdk is not installed. Jev semantic routing disabled.")


@dataclass
class JevDecision:
    is_addressing_bot: bool = False
    should_reply: bool = False
    tone: str = "normal"  # "normal" or "banter"
    confidence: float = 1.0
    is_fallback: bool = False


class JevService:
    def __init__(self):
        self._api_key = settings.typesafe_api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self._client: Optional[object] = None

    @property
    def is_configured(self) -> bool:
        return bool(TYPESAFE_AVAILABLE and (self._api_key or settings.typesafe_api_key))

    def _get_api_key(self) -> str:
        return settings.typesafe_api_key or self._api_key or os.environ.get("TYPESAFE_API_KEY", "")

    async def evaluate_message(
        self,
        text: str,
        sender_name: str = "เพื่อน",
        bot_name: str = "Thomas",
        nicknames: Optional[List[str]] = None,
        recent_context: str = "",
        quoted_context: str = "",
        timeout_seconds: float = 1.5
    ) -> JevDecision:
        """
        Evaluates a message using TypeSafe Jev System One model.
        Guarded by strict timeout so it never slows down the chatbot pipeline.
        """
        api_key = self._get_api_key()
        if not TYPESAFE_AVAILABLE or not api_key:
            return JevDecision(is_fallback=True)

        clean_text = text.strip()
        if not clean_text:
            return JevDecision(is_fallback=True)

        all_names = [bot_name] + (nicknames or [])

        state = {
            "message": clean_text,
            "sender": sender_name,
            "bot_name": bot_name,
            "bot_nicknames": all_names,
        }
        if recent_context:
            state["recent_chat_history"] = recent_context[-800:]
        if quoted_context:
            state["quoted_context"] = quoted_context[-400:]

        try:
            async def _call_jev() -> JevDecision:
                async with AsyncTypeSafeClient(api_key=api_key) as client:
                    resp = await client.system_one(
                        state=state,
                        questions={
                            "is_addressing_bot": Noul(
                                instructions=(
                                    f"Is the user directly calling, tagging, asking, or addressing the bot '{bot_name}' "
                                    f"(or its nicknames: {', '.join(all_names)}) in this message?"
                                )
                            ),
                            "should_reply": Noul(
                                instructions=(
                                    f"In a LINE group chat, should the bot '{bot_name}' reply to this message? "
                                    "Return yes if the message is directed to the bot, asks a question, or invites conversation."
                                )
                            ),
                            "tone": Choice(
                                instructions="What tone style is appropriate for answering this message?",
                                criteria={
                                    "normal": "Factual questions, help, technical questions, advice, serious discussion, or ordinary chatting",
                                    "banter": "Jokes, teasing, shitposting, memes, slang banter, roasting, or emotional banter"
                                }
                            )
                        }
                    )

                    addr_prob = resp.nouls["is_addressing_bot"].noul
                    should_prob = resp.nouls["should_reply"].noul
                    tone_choice = resp.choices["tone"].choice or "normal"

                    is_addr = addr_prob >= 0.58
                    should_rep = is_addr or should_prob >= 0.52

                    logger.info(
                        "Jev System One decision: addr=%.2f, should=%.2f, tone=%s for text: '%s'",
                        addr_prob, should_prob, tone_choice, clean_text[:40]
                    )

                    return JevDecision(
                        is_addressing_bot=is_addr,
                        should_reply=should_rep,
                        tone=tone_choice,
                        confidence=resp.choices["tone"].confidence or 1.0,
                        is_fallback=False
                    )

            return await asyncio.wait_for(_call_jev(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            logger.warning("TypeSafe Jev evaluation timed out after %.1fs, falling back to local heuristics", timeout_seconds)
            return JevDecision(is_fallback=True)
        except Exception as e:
            logger.warning("TypeSafe Jev evaluation failed (%s), falling back to local heuristics", e)
            return JevDecision(is_fallback=True)


jev_service = JevService()
