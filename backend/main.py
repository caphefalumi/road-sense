from pathlib import Path
import io
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Use relative paths from project root
PROJECT_ROOT = Path(__file__).parent.parent
CLS_MODEL_PATH = str(PROJECT_ROOT / "models" / "rubbish_cls" / "best.pt")
SEG_MODEL_PATH = str(PROJECT_ROOT / "models" / "rubbish_seg" / "best.pt")
CLS_THRESH     = 0.93
SEG_CONF       = 0.10  

CLASS_COLORS = [
    "hsl(200,80%,55%)", "hsl(260,60%,60%)", "hsl(170,70%,45%)",
    "hsl(30,80%,55%)",  "hsl(340,70%,55%)", "hsl(120,50%,45%)",
    "hsl(50,80%,50%)",  "hsl(290,60%,55%)", "hsl(10,75%,55%)",
    "hsl(160,65%,45%)", "hsl(220,70%,60%)", "hsl(0,70%,55%)",
]

CLASS_NAMES = [
    "Abandoned_shopping_cart", "Appliance", "Cardboard_box", "Furniture",
    "Garbage_bag", "Leftover_tire", "Mattress", "Metal_scrap",
    "Toy", "Trash", "Trash_pile", "Wooden_crate",
]

app = FastAPI(title="Rubbish Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
cls_model = None
seg_model = None

@app.on_event("startup")
def load_models():
    global cls_model, seg_model
    from ultralytics import YOLO
    print("Loading classifier...")
    cls_model = YOLO(CLS_MODEL_PATH)
    print("Loading segmenter...")
    seg_model = YOLO(SEG_MODEL_PATH)
    print("Models loaded.")

# Health check
@app.get("/api/health")
def health():
    return {"status": "ok", "cls": CLS_MODEL_PATH, "seg": SEG_MODEL_PATH}

# Main predict endpoint
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_w, img_h = img.size

    # Stage 1: Classifier
    cls_results = cls_model.predict(
        source=img, imgsz=224, verbose=False, device=DEVICE
    )
    probs      = cls_results[0].probs
    cls_names  = cls_results[0].names           # {0: 'notrubbish', 1: 'rubbish'}
    top1_id    = int(probs.top1)
    top1_conf  = float(probs.top1conf)
    top1_label = cls_names[top1_id]             # 'rubbish' or 'notrubbish'

    # Only skip segmentation if classifier is highly confident it's clean
    skip_seg = (top1_label != "rubbish") and (top1_conf >= CLS_THRESH)
    if skip_seg:
        return {
            "classifier": {"label": "notrubbish", "confidence": round(top1_conf, 4)},
            "objects": [],
        }

    # Stage 2: Segmentation (always runs when classifier is uncertain or says rubbish)
    seg_results = seg_model.predict(
        source=img, conf=SEG_CONF, iou=0.4, verbose=False, device=DEVICE
    )
    r     = seg_results[0]
    boxes = r.boxes

    objects = []
    for i, box in enumerate(boxes):
        cls_id  = int(box.cls)
        conf    = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bbox_norm = [
            round(x1 / img_w, 4),
            round(y1 / img_h, 4),
            round(x2 / img_w, 4),
            round(y2 / img_h, 4),
        ]

        # Extract mask polygon points (normalized to [0,1])
        polygon = None
        if r.masks is not None and i < len(r.masks.xy):
            pts = r.masks.xy[i]           # numpy array shape (N, 2) in pixel coords
            if len(pts) >= 3:
                polygon = [
                    [round(float(x) / img_w, 4), round(float(y) / img_h, 4)]
                    for x, y in pts
                ]

        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        objects.append({
            "label":      label,
            "confidence": round(conf, 4),
            "bbox":       bbox_norm,
            "polygon":    polygon,
            "color":      color,
        })

    # Render annotated image with masks (base64) for overlay display
    import base64, numpy as np
    annotated_bgr = r.plot(labels=True, boxes=True, masks=True)   # numpy BGR
    annotated_rgb = annotated_bgr[:, :, ::-1]                      # BGR → RGB
    pil_out  = Image.fromarray(annotated_rgb)
    buf      = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=88)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # Determine effective label: if seg found objects, treat as rubbish regardless of classifier
    effective_label = "rubbish" if (top1_label == "rubbish" or len(objects) > 0) else "notrubbish"

    return {
        "classifier":      {"label": effective_label, "confidence": round(top1_conf, 4)},
        "objects":         objects,
        "annotated_image": annotated_b64,
    }


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
