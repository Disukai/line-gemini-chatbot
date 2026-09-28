# 🤖 LINE Gemini Chatbot (เพื่อนซี้ AI ประจำกลุ่มไลน์)

> **LINE Chatbot AI ที่ขับเคลื่อนด้วย Google Gemini 3.8 Flash** ออกแบบมาสำหรับคุยในกลุ่มไลน์ส่วนตัวโดยเฉพาะ ตอบสนองอย่างเป็นธรรมชาติเหมือนเพื่อนมนุษย์จริงๆ ในกลุ่ม ไม่สแปม จำบริบทการคุยได้ และมาพร้อมคำสั่งพิเศษ `/plan`, `/boost`, `/goal` 🚀

---

## ✨ จุดเด่นและความสามารถ (Key Features)

1. **มนุษย์เพื่อน 100% ไม่ใช่หุ่นยนต์ Call Center**:
   - พิมพ์ภาษากลุ่มเพื่อนในไลน์อย่างเป็นธรรมชาติ (ฮะ, ครับ, วะ, เว้ย, 5555, เนอะ, ปะล่ะ)
   - ตอบสั้นกระชับ จังหวะจะโคนเหมือนคนกดแป้นพิมพ์มือถือจริง
   - รู้ว่าใครกำลังคุยกับใคร โดยดึงชื่อ Display Name ของเพื่อนแต่ละคนมาประกอบบริบท
2. **ไม่สแปมกลุ่ม (Smart Group Trigger)**:
   - บอทจะไม่พูดแทรกทุกข้อความจนกลุ่มพัง!
   - ตอบเมื่อ: ถูกแท็ก (@ชื่อ), เรียกชื่อบอท ("จิมมี่", "บอท", "เจมี่"), ตอบกลับ (Quote/Reply) ข้อความของบอท, หรือพิมพ์คำสั่งขึ้นต้นด้วย `/`
   - แอบเก็บประวัติการคุยเงียบๆ ในหน่วยความจำ (Sliding Window) ทำให้เมื่อถูกเรียก จะเข้าใจเรื่องที่เพื่อนๆ เพิ่งคุยกันทันที
3. **คำสั่งพิเศษจัดเต็ม (/Commands)**:
   - `/plan [หัวข้อ/เป้าหมาย]` : สวมบท Project Lead วาง Action Plan ละเอียดจัดเต็ม ขั้นตอน, จุดตาย, และ Quick-win ใน 15 นาทีแรก
   - `/boost [เรื่อง/ปัญหา]` : บูสต์พลังใจ ปลุกไฟให้ฮึกเหิม พร้อมทริคทะลวงจุดตันสไตล์เพื่อนแท้
   - `/goal [เป้าหมาย]` : แปลงเป้าหมายเป็น SMART Goal ชัดเจน + รายการสิ่งที่ต้องทำวันนี้ 3 ข้อ
   - `/persona [friend|chill|expert|snarky]` : ปรับเปลี่ยนโทนการคุย (เพื่อนซี้ / สายชิลล์ / มือโปร / สายกวน)
   - `/reset` : ล้างความจำบริบทการคุยของกลุ่มนี้
   - `/help` : แสดงเมนูช่วยเหลือและคำสั่งทั้งหมด
4. **รองรับรูปภาพ (Multimodal Image Understanding)**:
   - เพื่อนส่งรูปอาหาร, สถานที่ท่องเที่ยว หรือภาพอะไรเข้ามาในแชท บอทสามารถดูและแซว/คอมเมนต์ได้ทันที
5. **ความเร็วสูง & ไม่หลุด Webhook Timeout**:
   - รองรับ `ShowLoadingAnimation` ขึ้นสถานะกำลังพิมพ์ "..." ในไลน์
   - ใช้ FastAPI BackgroundTasks ตอบ `200 OK` กลับ LINE ภายในเสี้ยววินาที ป้องกันปัญหา LINE ส่งข้อความซ้ำ

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
line-gemini-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py           # การตั้งค่า Environment และ Settings
│   ├── persona.py          # Persona Engine & Prompt templates สำหรับภาษาไทย
│   ├── memory.py           # ระบบ Sliding Window Memory และ Profile Cache
│   ├── gemini_client.py    # ตัวเชื่อมต่อ Gemini 3.8 Flash (google-genai SDK)
│   ├── commands.py         # ตัวจัดการคำสั่ง /plan, /boost, /goal, /persona, /reset
│   ├── bot.py              # Logic ตรวจสอบเงื่อนไขการตอบ และ LINE API Dispatcher
│   └── main.py             # FastAPI Server & Webhook Verification
├── tests/
│   ├── __init__.py
│   └── test_bot.py         # Unit Tests ครอบคลุมการทำงานทุกโมดูล
├── scripts/
│   ├── run_local.sh        # สคริปต์รันเซิร์ฟเวอร์แบบง่าย
│   └── test_webhook.py     # ตัวจำลองส่งข้อความทดสอบโดยไม่ต้องต่อ LINE จริง
├── Dockerfile              # Docker Container
├── docker-compose.yml      # Docker Compose setup
├── requirements.txt        # Python Dependencies
├── pyproject.toml          # Package configuration & Pytest config
├── .env.example            # ตัวอย่างการตั้งค่า Environment Variables
└── README.md               # คู่มือการใช้งานนี้
```

---

## 🚀 ขั้นตอนการติดตั้งและตั้งค่าอย่างละเอียด (Step-by-Step Guide)

### 1. ขอ Gemini API Key (ฟรี)
1. ไปที่ [Google AI Studio](https://aistudio.google.com/)
2. ล็อกอินด้วยบัญชี Google แล้วคลิก **"Get API key"** -> **"Create API key"**
3. คัดลอก API Key เก็บไว้

### 2. สร้าง LINE Messaging API Channel
1. เข้าไปที่ [LINE Developers Console](https://developers.line.biz/console/)
2. สร้าง **Provider** (หรือเลือก Provider เดิมที่มีอยู่)
3. คลิก **"Create a new channel"** แล้วเลือก **"Messaging API"**
4. กรอกข้อมูลทั่วไป เช่น Channel name (ชื่อบอท เช่น `Jimmy AI`), Category, Email ให้เรียบร้อย
5. ไปที่แท็บ **"Messaging API"**:
   - เลื่อนลงมาที่หัวข้อ **Channel access token** แล้วกด **"Issue"** เพื่อรับ Token ระยะยาว (Long-lived)
6. ไปที่แท็บ **"Basic settings"**:
   - เลื่อนหา **Channel secret** แล้วกดคัดลอกไว้

### 3. ตั้งค่าให้บอทเข้ากลุ่มไลน์ได้ (สำคัญมาก! ⚠️)
1. ในหน้า [LINE Developers Console](https://developers.line.biz/console/) แท็บ **Messaging API**
2. เลื่อนลงมาที่หัวข้อ **"LINE Official Account features"** แล้วคลิก Edit ที่ **"Auto-reply messages"** (จะเปิดหน้า LINE Official Account Manager)
3. ในหน้าการตั้งค่าการตอบกลับ:
   - **ข้อความตอบกลับอัตโนมัติ (Auto-response)**: ปิด (Disabled) ❌ *(ถ้าเปิด LINE จะตอบข้อความเริ่มต้นเอง)*
   - **ข้อความทักทายเพื่อนใหม่ (Greeting message)**: ปิด (Disabled) ❌ *(ถ้าไม่ต้องการให้ทักซ้ำซ้อน)*
   - **Webhook**: เปิด (Enabled) ✅
4. ไปที่หน้า **การตั้งค่าบัญชี (Account settings)** -> แถบ **แชท (Chat)**:
   - **อนุญาตให้เข้าร่วมการสนทนาแบบกลุ่มหรือแบบหลายคน (Allow account to join groups and multi-person chats)**: เลือก **เปิด (Enabled)** ✅

---

### 4. การรันเซิร์ฟเวอร์ในเครื่อง (Local Setup)

#### ก. คัดลอกและตั้งค่า `.env`
```bash
cd line-gemini-chatbot
cp .env.example .env
```
เปิดไฟล์ `.env` แล้วใส่ค่าคีย์ที่คุณได้มา:
```env
LINE_CHANNEL_ACCESS_TOKEN="ใส่ Channel Access Token ที่ได้จาก LINE"
LINE_CHANNEL_SECRET="ใส่ Channel Secret ที่ได้จาก LINE"
GEMINI_API_KEY="ใส่ Gemini API Key จาก Google AI Studio"
GEMINI_MODEL="gemini-3.8-flash"
BOT_NAME="จิมมี่"
```

#### ข. ติดตั้ง Dependencies และรัน
คุณสามารถใช้ script อัตโนมัติ:
```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```
หรือรันด้วยมือ:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
เมื่อรันสำเร็จ หน้าจอจะแสดง:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### ค. รันการทดสอบ Unit Tests
```bash
source .venv/bin/activate
pytest tests/
```

---

### 5. เชื่อมต่อ Webhook กับ LINE (Webhook Configuration)

LINE ต้องการ Webhook ที่เป็น **HTTPS**:

#### ตัวเลือกที่ 1: ใช้ Cloudflare Tunnel (ฟรี & เสถียร ไม่เปลี่ยน URL บ่อย)
```bash
cloudflared tunnel --url http://localhost:8000
```

#### ตัวเลือกที่ 2: ใช้ Ngrok (ง่ายและเร็ว)
```bash
ngrok http 8000
```
คุณจะได้ Forwarding URL เช่น `https://xxxx-xx-xx-xx.ngrok-free.app`

#### นำ URL ไปใส่ใน LINE Console:
1. ไปที่ [LINE Developers Console](https://developers.line.biz/console/) -> เลือก Channel ของคุณ -> แท็บ **Messaging API**
2. ในหัวข้อ **Webhook URL** กด **Edit**:
   - ใส่: `https://xxxx-xx-xx-xx.ngrok-free.app/callback`
   - กด **Update**
3. กดปุ่ม **"Verify"**:
   - หากขึ้น **Success** แสดงว่าการเชื่อมต่อและ Signature Verification สมบูรณ์แบบ!
4. เปิดสวิตช์ **"Use webhook"** ให้เป็นสีเขียว (ON)

---

### 6. การทดสอบในเครื่องโดยไม่ต้องรอต่อ LINE (Offline Simulation)

เรามีสคริปต์ `scripts/test_webhook.py` ให้คุณสามารถยิงข้อความจำลองทดสอบการทำงานของบอทได้ทันที:

```bash
# ทดสอบคุยธรรมดาในกลุ่ม
python3 scripts/test_webhook.py "จิมมี่ วันนี้กินอะไรกันดี"

# ทดสอบคำสั่ง /plan
python3 scripts/test_webhook.py "/plan เตรียมอ่านสอบใน 14 วัน"

# ทดสอบคำสั่ง /boost
python3 scripts/test_webhook.py "/boost ท้อแท้กับงานมากเลยเพื่อน"
```

---

## 💬 ตัวอย่างบทสนทนาจริงในกลุ่มไลน์

### ตัวอย่างที่ 1: คุยเล่นทั่วไป
> **เพื่อนในกลุ่ม (Harvey):** เย็นนี้เลิกงานไปไหนกันดีวะ  
> **เพื่อนในกลุ่ม (Pluem):** หมูกระทะมั้ย  
> **เพื่อนในกลุ่ม (Harvey):** จิมมี่ คิดว่าร้านไหนเด็ดสุดแถวสยาม  
> **บอท (จิมมี่):** ถ้าสยามต้องนักล่าหมูกระทะชั้นใต้ดินเลยพี่ฮาร์วีย์ ติดแอร์ ไม่ร้อนด้วย แต่ไปไวหน่อยนะ คนแน่นตลอด 555

### ตัวอย่างที่ 2: คำสั่ง `/plan`
> **เพื่อนในกลุ่ม:** `/plan วางแผนเปิดคอร์สสอนออนไลน์เรื่อง AI ภายใน 1 เดือน`  
> **บอท (จิมมี่):**  
> 🎯 **1. เป้าหมาย & นิยามความสำเร็จ (Target Outcome)**  
> ปล่อยคอร์สออนไลน์ AI ตัวแรกพร้อมระบบรับชำระเงินและมีนักเรียนรุ่นแรกอย่างน้อย 20 คนภายใน 30 วัน  
>  
> 📋 **2. แผนปฏิบัติการแบบ Step-by-Step (Action Steps & Milestones)**  
> - **Week 1 (วางโครง & ทำสไลด์):** สรุป Outlines 5 โมดูลหลักที่คนอยากเรียนจริงๆ เน้น Workshop ไม่เน้นทฤษฎี  
> - **Week 2 (บันทึกวิดีโอ & ตัดต่อ):** อัดคลิปสั้นบทละ 10-15 นาที ตัดต่อให้กระชับ  
> - **Week 3 (ทำ Landing Page & Early Bird):** เปิดพรีออเดอร์ในกลุ่มเป้าหมายราคาพิเศษ ดึงฟีดแบ็ก  
> - **Week 4 (Go Live & Follow-up):** ส่งมอบคอร์ส ตั้งกลุ่ม Discord/LINE ดูแลนักเรียน  
>  
> ⚠️ **3. จุดตาย & สิ่งที่ต้องระวัง (Pitfalls & Mitigation)**  
> - ระวังทำเนื้อหากว้างเกินไปจนอัดคลิปไม่ทัน -> โฟกัสปัญหาเดียวที่คนยอมจ่ายเงิน  
>  
> 🚀 **4. Next 15 Minutes Quick-Win**  
> หยิบกระดาษมาเขียน 3 ปัญหาหลักที่คอร์สนี้จะช่วยแก้ให้คนเรียนให้เสร็จตอนนี้เลยเพื่อน จัดไป!

---

## 🐳 การรันด้วย Docker Compose

หากคุณต้องการรันแบบ Background Service หรือ Deploy ขึ้น VPS:

```bash
# ตรวจสอบว่ามีไฟล์ .env เรียบร้อย
docker compose up -d --build

# ดู Log การทำงาน
docker compose logs -f
```

---

## ☁️ การ Deploy ขึ้น Cloud แบบ 24/7 (แนะนำ)

- **Render.com / Railway.app**:
  1. Push โค้ดนี้ขึ้น GitHub (Private Repo)
  2. สร้าง Web Service ใหม่ แล้วเชื่อมกับ GitHub Repository
  3. ใส่ Environment Variables (`LINE_CHANNEL_ACCESS_TOKEN`, `LINE_CHANNEL_SECRET`, `GEMINI_API_KEY`)
  4. นำ URL ที่ได้ (เช่น `https://my-line-bot.onrender.com/callback`) ไปใส่ใน LINE Developers Console

---

## 🛡️ License & Maintenance
พัฒนาด้วยความใส่ใจเพื่อประสบการณ์การสนทนาที่เป็นธรรมชาติที่สุดบน LINE โดยใช้เทคโนโลยี Gemini 3.8 Flash ⚡
