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
        default="gemini-3.1-flash-lite",
        description="Gemini Model identifier (e.g. gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-lite-latest)"
    )

    # Bot Identity & Persona
    bot_name: str = Field(
        default="Thomas",
        description="Bot's primary name in the group chat"
    )
    bot_nicknames_str: str = Field(
        default="thomas,โทมัส,โธมัส,ทอม,ทอมมี่,บอท,ai,เพื่อน,จิมมี่",
        description="Comma-separated aliases or nicknames for the bot"
    )
    default_persona: str = Field(
        default="friend",
        description="Default persona style: friend, chill, expert, or snarky"
    )

    # Group Chat Behavior
    # Options: "chime_in" (mention + natural spontaneous chime-ins),
    #          "mention" (reply only when tagged/named/replied to/command),
    #          "smart" (mention + intelligent chime-in on group questions),
    #          "all" (reply to every message in group - not recommended)
    group_trigger_mode: str = Field(
        default="chime_in",
        description="Trigger behavior in group chats: chime_in, mention, smart, or all"
    )
    spontaneous_base_rate: float = Field(
        default=0.25,
        description="Base probability (0.0 to 1.0) for spontaneous chime-in in group chats"
    )
    spontaneous_question_bonus: float = Field(
        default=0.20,
        description="Bonus probability added when message contains questions or requests for opinion"
    )
    spontaneous_slang_bonus: float = Field(
        default=0.15,
        description="Bonus probability added when message contains slang, laughter, or strong emotion"
    )
    spontaneous_image_rate: float = Field(
        default=0.30,
        description="Probability for spontaneous reaction when a photo is posted in group chat"
    )
    spontaneous_cooldown_seconds: int = Field(
        default=25,
        description="Minimum seconds between spontaneous chime-ins in a group"
    )
    spontaneous_min_messages: int = Field(
        default=1,
        description="Minimum non-bot messages required before another spontaneous chime-in"
    )
    max_history_per_group: int = Field(
        default=25,
        description="Maximum recent conversation messages kept in memory per group"
    )
    enable_loading_animation: bool = Field(
        default=True,
        description="Show 'typing/loading...' animation in LINE while generating response"
    )

    # Cloud Keep-Alive Self-Ping (Prevents Render Free Tier Cold Starts / Sleep)
    enable_keep_alive: bool = Field(
        default=True,
        description="Whether to periodically ping the public health endpoint to prevent idle container sleep"
    )
    keep_alive_url: str = Field(
        default="https://line-gemini-thomas.onrender.com/health",
        description="Public URL to ping for keeping the cloud container alive"
    )
    keep_alive_interval_seconds: int = Field(
        default=480,
        description="Interval in seconds between keep-alive pings (8 minutes)"
    )

    # Autonomous Proactive Schedulers
    enable_proactive_chatter: bool = Field(
        default=True,
        description="Enable periodic spontaneous conversation openers in active groups"
    )
    chatter_interval_min_hours: float = Field(
        default=3.5,
        description="Minimum hours between random conversation openers"
    )
    chatter_interval_max_hours: float = Field(
        default=5.0,
        description="Maximum hours between random conversation openers"
    )
    enable_morning_news: bool = Field(
        default=True,
        description="Enable daily morning news briefing broadcast to active groups"
    )
    morning_news_hour: int = Field(
        default=8,
        description="Hour of day (0-23 in Bangkok time UTC+7) to send morning news briefing"
    )
    morning_news_minute: int = Field(
        default=0,
        description="Minute of hour (0-59 in Bangkok time UTC+7) to send morning news briefing"
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
    def effective_group_trigger_mode(self) -> str:
        """
        Promotes legacy 'mention' setting to 'chime_in' unless overridden per-chat via /mode,
        ensuring the bot naturally chimes in on questions and banter.
        """
        if self.group_trigger_mode in ("chime_in", "smart", "all"):
            return self.group_trigger_mode
        return "chime_in"

    @property
    def candidate_models(self) -> List[str]:
        """Ordered list of Gemini model candidates for resilient fallback."""
        base = [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            self.gemini_model,
            "gemini-3.6-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest"
        ]
        seen = set()
        res = []
        for m in base:
            if m and m not in seen:
                seen.add(m)
                res.append(m)
        return res


settings = Settings()
