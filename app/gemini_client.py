"""
Gemini Client module using the official google-genai SDK.
Primary target: gemini-3.8-flash with automatic multi-model candidate fallback resilience.
Supports text reasoning, group context awareness, and multimodal image input.
"""
import logging
from typing import Optional, List, Union

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger("line_gemini_bot")


class GeminiClient:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self._working_model: Optional[str] = None
        self._init_client()

    def _init_client(self):
        if settings.gemini_api_key:
            try:
                self._client = genai.Client(api_key=settings.gemini_api_key)
                self._working_model = settings.gemini_model
                logger.info("Gemini Client initialized with target model: %s", settings.gemini_model)
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
        mime_type: str = "image/jpeg"
    ) -> str:
        """
        Generates a human-like response from Gemini 3.8 Flash (with automatic candidate fallback).
        Incorporates group conversation history and handles text or image input.
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

        if image_bytes:
            try:
                prompt_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            except Exception as img_err:
                logger.warning("Failed to create image Part: %s", img_err)

        # Frame the context cleanly
        context_block = ""
        if history_context:
            context_block = f"--- ประวัติการคุยล่าสุดในกลุ่ม ---\n{history_context}\n--- จบประวัติ ---\n\n"

        current_turn = f"{context_block}[{sender_name}]: {user_message}"
        prompt_parts.append(current_turn)

        # Build candidate models, prioritizing the last confirmed working model
        candidates = list(settings.candidate_models)
        if self._working_model and self._working_model in candidates:
            candidates.remove(self._working_model)
            candidates.insert(0, self._working_model)

        last_error = None

        for model_name in candidates:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.8,
                    max_output_tokens=300,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
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
                        return "เรื่องนี้ระบบแจ้งว่าติดฟิลเตอร์ความปลอดภัยแฮะเพื่อน 555 ขอผ่านก่อนนะ ลองเปลี่ยนประเด็นคุยดู!"

                if response and response.text:
                    self._working_model = model_name
                    return response.text.strip()
            except Exception as gen_err:
                logger.warning("generate_content failed on model %s: %s", model_name, gen_err)
                last_error = gen_err

        logger.error("All Gemini model candidates failed. Last error: %s", last_error)
        return "แป๊บนะเพื่อน สมองเบลอชั่วคราว มีบั๊กจากฝั่ง API ลองทักมาใหม่อีกรอบดิ๊ 555"


gemini_client = GeminiClient()
