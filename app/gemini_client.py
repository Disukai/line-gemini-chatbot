"""
Gemini Client module using the official google-genai SDK.
Primary target: gemini-3.8-flash with automatic multi-model candidate fallback resilience.
Supports text reasoning, group context awareness, and multimodal image input.
"""
import time
import logging
from typing import Optional, List, Union, Dict

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger("line_gemini_bot")


class GeminiClient:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self._working_model: Optional[str] = None
        self._bad_models: Dict[str, float] = {}  # {model_name: blacklisted_until_epoch}
        self._init_client()

    def _init_client(self):
        if settings.gemini_api_key:
            try:
                # Configure attempts=1 so 429/404 fail in <0.3s rather than wasting 30+ seconds retrying
                self._client = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options=types.HttpOptions(
                        retry_options=types.HttpRetryOptions(attempts=1)
                    )
                )
                self._working_model = None
                self._bad_models = {}
                logger.info("Gemini Client initialized with zero-wait fast failover and multi-model candidate fallback")
            except Exception as e:
                logger.error("Failed to initialize Gemini Client: %s", e)
                self._client = None
        else:
            logger.warning("GEMINI_API_KEY is not set. Gemini calls will return placeholder notices.")

    @property
    def is_configured(self) -> bool:
        return bool(self._client and settings.gemini_api_key)

    def generate_chat_response(
        self,
        user_message: str,
        system_instruction: str,
        history_context: str = "",
        sender_name: str = "เพื่อน",
        image_bytes: Optional[bytes] = None,
        mime_type: str = "image/jpeg",
        media_bytes: Optional[bytes] = None,
        media_mime_type: Optional[str] = None
    ) -> str:
        """
        Generates a human-like response from Gemini 3.8 Flash (with automatic candidate fallback).
        Incorporates group conversation history and handles text, document, or image input.
        """
        if not self.is_configured:
            # Re-attempt init in case env var was updated at runtime
            self._init_client()
            if not self.is_configured:
                return (
                    "เฮ้ยเพื่อน ลืมใส่ GEMINI_API_KEY ในไฟล์ .env หรือเปล่า! "
                    "ไปเอาคีย์ฟรีที่ aistudio.google.com มาใส่ก่อนนะ แล้วค่อยคุยกันใหม่ 555"
                )

        # Construct conversational input
        prompt_parts: List[Union[str, types.Part]] = []

        # Handle multimodal data: accept either media_bytes or image_bytes
        data_to_send = media_bytes if media_bytes is not None else image_bytes
        target_mime = media_mime_type or mime_type

        if data_to_send:
            try:
                prompt_parts.append(types.Part.from_bytes(data=data_to_send, mime_type=target_mime))
            except Exception as media_err:
                logger.warning("Failed to create media Part: %s", media_err)

        # Frame the context cleanly
        context_block = ""
        if history_context:
            context_block = f"--- ประวัติการคุยล่าสุด ---\n{history_context}\n--- จบประวัติ ---\n\n"

        current_turn = f"{context_block}[{sender_name}]: {user_message}"
        prompt_parts.append(current_turn)

        # Build candidate models, filtering out models under temporary cooldown
        now = time.time()
        self._bad_models = {m: exp for m, exp in self._bad_models.items() if exp > now}

        candidates = [m for m in settings.candidate_models if m not in self._bad_models]
        if not candidates:
            # If all candidates are cooling down, reset and try all
            candidates = list(settings.candidate_models)

        if self._working_model and self._working_model in candidates:
            candidates.remove(self._working_model)
            candidates.insert(0, self._working_model)

        last_error = None

        for model_name in candidates:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.75,
                    max_output_tokens=2048
                )
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt_parts,
                    config=config
                )

                # Check if candidates were blocked by safety filters
                if response and hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason and str(finish_reason).upper() in ["SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"]:
                        return "อันนี้ติดฟิลเตอร์ความปลอดภัยเฉยเลย 5555555 ขอผ่านก่อนนะ ลองเปลี่ยนเรื่องคุยดู!"

                if response and response.text:
                    self._working_model = model_name
                    return response.text.strip()
            except Exception as gen_err:
                err_str = str(gen_err)
                logger.warning("generate_content failed on model %s: %s", model_name, gen_err)
                last_error = gen_err
                if "429" in err_str:
                    # Quota exhausted: blacklist for 10 minutes to avoid repeated failover delay
                    self._bad_models[model_name] = now + 600
                    if self._working_model == model_name:
                        self._working_model = None
                elif "404" in err_str:
                    # Deprecated / unavailable: blacklist for 1 hour
                    self._bad_models[model_name] = now + 3600
                    if self._working_model == model_name:
                        self._working_model = None
                elif "503" in err_str:
                    # Temporary demand spike: blacklist for 1 minute
                    self._bad_models[model_name] = now + 60
                    if self._working_model == model_name:
                        self._working_model = None

        logger.error("All Gemini model candidates failed. Last error: %s", last_error)
        return "แป๊บนะมึง สมองเบลอชั่วคราว ฝั่ง API กำลัง cooked ทักมาใหม่อีกทีดิ๊ 5555555"


gemini_client = GeminiClient()
