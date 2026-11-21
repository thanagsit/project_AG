import os
import sys
import uuid
import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    ImageMessage as ReplyImageMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)

from ultralytics import YOLO

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# ========== IMPORT PIPELINE COMPONENTS ==========
# Make sure we can import from projectAG_New root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model
from data import get_val_transforms
from utils import load_checkpoint

# ================================================
# 1. SETUP CONFIG AND MODELS
# ================================================
app = FastAPI(title="Pig Weight Predictor Bot")

from dotenv import load_dotenv
load_dotenv()

# --- LINE API Credentials ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    print("ERROR: LINE_CHANNEL_ACCESS_TOKEN or LINE_CHANNEL_SECRET is missing in .env file!")
    sys.exit(1)

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- Define Directories ---
# Static directory to host images temporarily so LINE can download them
STATIC_DIR = Path("static_images")
STATIC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOWNLOADS_DIR = STATIC_DIR / "raw"
RAW_DOWNLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = STATIC_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Mount static folder so images are accessible via HTTP
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Load Models Manually into VRAM ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initialize AI on Device: {DEVICE}")

# --- Load YOLO Pig Detector ---
YOLO_MODEL_PATH = Path("backup_weights/bestv8.pt")
yolo_model = None
if YOLO_MODEL_PATH.exists():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO Pig Detector loaded successfully!")
else:
    print(f"⚠️ Warning: YOLO model not found at {YOLO_MODEL_PATH}")

# Weight Prediction Models (ConvNeXt Small)
WEIGHT_MODEL_DIR = Path("outputs/checkpoints/convnext_small")
if not WEIGHT_MODEL_DIR.exists() or len(list(WEIGHT_MODEL_DIR.glob("best_fold*.pth"))) == 0:
    print(f"Warning: ConvNeXt weights missing in {WEIGHT_MODEL_DIR}")

CONFIG_PATH = "config/config.yaml"
cfg = OmegaConf.load(CONFIG_PATH)
cfg.model.backbone = "convnext_small" 
transform = get_val_transforms(cfg.data.img_size)

print("Loading Weight Models (Ensemble)...")
weight_models = []
if WEIGHT_MODEL_DIR.exists():
    for fold_path in sorted(list(WEIGHT_MODEL_DIR.glob("best_fold*.pth"))):
        w_model = build_model(cfg)
        load_checkpoint(w_model, str(fold_path), device=DEVICE)
        w_model.to(DEVICE)
        w_model.eval()
        weight_models.append(w_model)
    print(f"Loaded Weight Models {len(weight_models)} Folds successfully!")
else:
    print("Error: Weight models not found.")

# ================================================
# 2. CORE PREDICTION LOGIC
# ================================================
def predict_weight(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad():
        batch = image_tensor.unsqueeze(0).to(device)
        pred = model(batch)
        return pred.item()

def process_pig_image(file_path: Path) -> dict:
    """ Processes raw image using ConvNeXt Ensemble on the full frame """
    orig_img = Image.open(file_path).convert("RGB")
    
    # Predict on Full Image
    image_tensor = transform(image=np.array(orig_img))["image"]
    
    preds = []
    for w_model in weight_models:
        preds.append(predict_weight(w_model, image_tensor, DEVICE))
        
    final_weight = sum(preds) / len(preds) if preds else 0.0
    
    message = f"ผลการทำนายน้ำหนัก: {final_weight:.2f} kg"
    
    return {
        "success": True, 
        "message": message
    }

# ================================================
# 3. FASTAPI ENDPOINTS & LINE HANDLERS
# ================================================

@app.get("/")
async def root():
    return {"message": "Pig Weight Predictor Bot is running!"}

@app.post("/callback")
async def callback(request: Request):
    """ LINE Webhook Endpoint """
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")
    
    try:
        handler.handle(body_str, signature)
    except InvalidSignatureError:
        print("Invalid Signature!")
        raise HTTPException(status_code=400, detail="Invalid signature. Check your channel secret.")
    except Exception as e:
        import traceback
        print("Error in handler:")
        traceback.print_exc()
        raise e
    
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    print(f"Received Text: {event.message.text} from {event.source.user_id}")
    """ 
    Handle text messages. 
    Just give instructions if they type words.
    """
    msg = "สวัสดีครับ ผมพร้อมทำนายน้ำหนักให้แล้วครับ\n\nเพียงส่งรูปหมูเข้ามา ผมจะรีบคำนวณและแจ้งน้ำหนักให้ทันทีเลยครับ!"
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=msg)]
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    print(f"Received Image from {event.source.user_id}")
    """ 
    Handle image messages -> Download -> Process -> Reply 
    """
    message_id = event.message.id
    
    # 1. Download image content from LINE servers
    filename = f"{uuid.uuid4().hex}.jpg"
    raw_path = RAW_DOWNLOADS_DIR / filename
    
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        line_bot_api = MessagingApi(api_client)
        
        message_content = line_bot_blob_api.get_message_content(message_id)
        with open(raw_path, 'wb') as fd:
            fd.write(message_content)
                
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="ได้รับรูปแล้วครับ ผมกำลังประมวลผลน้ำหนัก")]
            )
        )
        
        # 2. Process Image (Run Detection First)
        user_id = event.source.user_id
        if yolo_model:
            # ใช้ conf ต่ำลงนิดนึง และรองรับทั้ง BBox และ Segmentation
            results = yolo_model.predict(source=str(raw_path), conf=0.15)
            
            # เช็คว่าเจออะไรไหม (เงื่อนไขครอบคลุมทั้ง Detect และ Seg)
            has_detection = len(results[0].boxes) > 0 or (hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0)
            
            if not has_detection:
                line_bot_api.push_message(
                    PushMessageRequest(
                        to=user_id,
                        messages=[TextMessage(text="❌ ไม่พบหมูในรูปที่ส่งมาครับ\nกรุณาส่งรูปหมูที่เห็นตัวชัดเจนเพื่อให้ผมช่วยคำนวณน้ำหนักนะครับ!")]
                    )
                )
                return

        # 3. Process Weight Prediction (Run AI)        
        process_result = process_pig_image(raw_path)
        
        # 4. Formulate Reply Push
        line_bot_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text=process_result["message"])]
            )
        )

# For running via python script instead of uvicorn CLI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
