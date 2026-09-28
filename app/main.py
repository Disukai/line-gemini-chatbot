"""
FastAPI application entrypoint for LINE Gemini Chatbot webhook.
Handles HTTP requests, LINE signature verification, and async background dispatching.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    FileMessageContent,
    JoinEvent,
    FollowEvent,
)

from app.config import settings
from app.bot import (
    process_text_message,
    process_image_message,
    process_file_message,
    process_join_event,
    process_follow_event,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("line_gemini_bot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("==================================================")
    logger.info("🚀 LINE Gemini Chatbot is starting up!")
    logger.info("🤖 Model: %s", settings.gemini_model)
    logger.info("👥 Bot Name: %s", settings.bot_name)
    logger.info("🎯 Group Trigger Mode: %s", settings.effective_group_trigger_mode)
    logger.info("⏰ Proactive Chatter: %s (Every %.1f-%.1f hrs)", settings.enable_proactive_chatter, settings.chatter_interval_min_hours, settings.chatter_interval_max_hours)
    logger.info("📰 Morning News: %s (%02d:%02d BKK time)", settings.enable_morning_news, settings.morning_news_hour, settings.morning_news_minute)
    logger.info("==================================================")
    from app.scheduler import scheduler
    scheduler.start()
    yield
    scheduler.stop()
    logger.info("🛑 LINE Gemini Chatbot shutting down.")


app = FastAPI(
    title="LINE Gemini Chatbot",
    description="Realistic Thai human-like AI chatbot for private LINE groups powered by Gemini 3.8 Flash",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "line-gemini-chatbot",
        "bot_name": settings.bot_name,
        "model": settings.gemini_model,
        "instructions": "Set your LINE Webhook URL to https://<your-domain>/callback"
    }


@app.get("/health")
async def health_check():
    from app.chat_tracker import chat_tracker
    return {
        "status": "healthy",
        "gemini_configured": bool(settings.gemini_api_key),
        "line_configured": bool(settings.line_channel_secret and settings.line_channel_access_token),
        "scheduler_enabled": settings.enable_proactive_chatter or settings.enable_morning_news,
        "active_groups_count": len(chat_tracker.get_active_group_ids())
    }


@app.post("/callback")
async def callback(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(None, alias="X-Line-Signature")
):
    """
    LINE Webhook callback endpoint.
    Verifies signature and processes events asynchronously via BackgroundTasks
    to ensure instant HTTP 200 response back to LINE within the 1-second timeout.
    """
    if not settings.line_channel_secret:
        logger.error("LINE_CHANNEL_SECRET is not configured!")
        raise HTTPException(status_code=500, detail="LINE_CHANNEL_SECRET is missing")

    if not x_line_signature:
        logger.warning("Missing X-Line-Signature header")
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_bytes = await request.body()
    body = body_bytes.decode("utf-8")

    events = None
    last_sig_err = None
    for sec in settings.candidate_secrets:
        try:
            parser = WebhookParser(sec)
            events = parser.parse(body, x_line_signature)
            break
        except InvalidSignatureError as err:
            last_sig_err = err
            continue
        except Exception as e:
            logger.error("Failed to parse LINE Webhook payload: %s", e)
            raise HTTPException(status_code=400, detail=f"Failed to parse payload: {e}")

    if events is None:
        logger.warning("Invalid LINE Webhook signature received!")
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, TextMessageContent):
                background_tasks.add_task(process_text_message, event)
            elif isinstance(event.message, ImageMessageContent):
                background_tasks.add_task(process_image_message, event)
            elif isinstance(event.message, FileMessageContent):
                background_tasks.add_task(process_file_message, event)
        elif isinstance(event, JoinEvent):
            background_tasks.add_task(process_join_event, event)
        elif isinstance(event, FollowEvent):
            background_tasks.add_task(process_follow_event, event)

    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
