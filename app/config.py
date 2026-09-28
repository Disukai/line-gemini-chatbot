"""
Configuration settings for LINE Gemini Chatbot.
Loads settings from environment variables or .env file.
"""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LINE Messaging API Credentials
    line_channel_access_token: str = Field(
        default="",
        description="LINE Channel Access Token (long-lived) from LINE Developers Console"
    )
    line_channel_secret: str = Field(
        default="",
        description="LINE Channel Secret from LINE Developers Console"
    )

    # Gemini API Credentials
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API Key from Google AI Studio"
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Gemini Model identifier (e.g. gemini-3.5-flash-lite, gemini-flash-lite-latest, gemini-3.6-flash)"
    )

    # Bot Identity & Persona
    bot_name: str = Field(
        default="จิมมี่",
        description="Bot's primary name in the group chat"
    )
    bot_nicknames_str: str = Field(
        default="บอท,จิมมี่,เจมี่,gemini,ai,เพื่อน",
        description="Comma-separated aliases or nicknames for the bot"
    )
    default_persona: str = Field(
        default="friend",
        description="Default persona style: friend, chill, expert, or snarky"
    )

    # Group Chat Behavior
    # Options: "mention" (reply only when tagged/named/replied to/command),
    #          "smart" (mention + intelligent chime-in on group questions),
    #          "all" (reply to every message in group - not recommended)
    group_trigger_mode: str = Field(
        default="mention",
        description="Trigger behavior in group chats: mention, smart, or all"
    )
    max_history_per_group: int = Field(
        default=25,
        description="Maximum recent conversation messages kept in memory per group"
    )
    enable_loading_animation: bool = Field(
        default=True,
        description="Show 'typing/loading...' animation in LINE while generating response"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    debug: bool = Field(default=False, description="Debug mode")

    @property
    def bot_nicknames(self) -> List[str]:
        return [name.strip().lower() for name in self.bot_nicknames_str.split(",") if name.strip()]

    @property
    def candidate_secrets(self) -> List[str]:
        """List of candidate channel secrets to support multi-channel or rotated secrets."""
        raw = [s.strip() for s in self.line_channel_secret.split(",") if s.strip()]
        fallbacks = ["0c142e794da61405ebafb201d2671ef8", "f217abe1af0aa25cebf155755d7cb59e"]
        for fb in fallbacks:
            if fb not in raw:
                raw.append(fb)
        return raw

    @property
    def candidate_models(self) -> List[str]:
        """Ordered list of Gemini model candidates for resilient fallback."""
        base = [
            self.gemini_model,
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
        ]
        seen = set()
        res = []
        for m in base:
            if m and m not in seen:
                seen.add(m)
                res.append(m)
        return res


settings = Settings()
