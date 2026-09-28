"""
Local test script to simulate LINE Webhook events.
Useful for testing the bot locally without needing ngrok or a live LINE connection.
"""
import base64
import hashlib
import hmac
import json
import sys
import httpx

# Default local server URL
SERVER_URL = "http://127.0.0.1:8000/callback"
TEST_SECRET = "test_channel_secret_12345"


def generate_signature(body: str, secret: str) -> str:
    """Computes HMAC-SHA256 signature for LINE webhook payload."""
    hash_value = hmac.new(
        secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256
    ).digest()
    return base64.b64encode(hash_value).decode("utf-8")


def send_simulated_message(text: str, is_group: bool = True, secret: str = TEST_SECRET):
    """Sends a simulated LINE text message event to the local server."""
    if is_group:
        source = {
            "type": "group",
            "groupId": "Ca9876543210group",
            "userId": "U1234567890harvey"
        }
    else:
        source = {
            "type": "user",
            "userId": "U1234567890harvey"
        }

    payload = {
        "destination": "Ubotdestination123",
        "events": [
            {
                "type": "message",
                "message": {
                    "type": "text",
                    "id": "100001",
                    "text": text
                },
                "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZR",
                "deliveryContext": {
                    "isRedelivery": False
                },
                "timestamp": 1720000000000,
                "source": source,
                "replyToken": "nHuyWiB7yP5Zw52FIkcQobQuGDXCTA",
                "mode": "active"
            }
        ]
    }

    body = json.dumps(payload)
    signature = generate_signature(body, secret)

    headers = {
        "Content-Type": "application/json",
        "X-Line-Signature": signature
    }

    print(f"\n📤 Sending simulated message: '{text}' (is_group={is_group})")
    try:
        response = httpx.post(SERVER_URL, content=body, headers=headers, timeout=10.0)
        print(f"📥 Server response: {response.status_code} - {response.text}")
    except httpx.ConnectError:
        print(f"❌ Could not connect to {SERVER_URL}. Make sure your local server is running!")


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "จิมมี่ วันนี้ไปกินอะไรกันดีวะ"
    send_simulated_message(msg)
