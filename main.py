import os
import io
import logging
import tempfile

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# ===== Config =====
# Uses path relative to this file — works on Windows, Linux, Docker
MODEL_PATH = "Final_best_model.pth"
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


# ===== Model Loading (lifespan) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Fail fast with a clear error if the model file is missing
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Make sure 'Final_best_model.pth' is in the same directory as main.py."
        )

    logger.info(f"Loading model from {MODEL_PATH} on device: {device}")

    model_local = deeplabv3_resnet101(weights=None, aux_loss=True)

    # Exact same custom head used during training
    model_local.classifier[4] = torch.nn.Sequential(
        torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, NUM_CLASSES, kernel_size=1),
    )
    model_local.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model_local.load_state_dict(state_dict)
    model_local.to(device)
    model_local.eval()

    model = model_local
    logger.info("Model loaded successfully and ready for inference.")

    yield  # App runs here

    model = None
    logger.info("Model unloaded on shutdown.")


# ===== FastAPI App =====
app = FastAPI(
    lifespan=lifespan,
    title="Nani AI — Segmentation API",
    version="1.0.0",
    description="DeepLabv3_ResNet101 semantic segmentation. Accepts a video file, "
                "segments the first frame, and returns contour + corner points.",
)


# ===== Helper: preprocess =====
def preprocess_image(pil_image: Image.Image):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_image).unsqueeze(0).to(device)


# ===== Helper: corner detection =====
def detect_real_corners(contour, w: int, h: int):
    contour_image = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_image, [contour], -1, 255, 2)
    corners = cv2.goodFeaturesToTrack(
        contour_image, maxCorners=8, qualityLevel=0.5, minDistance=50
    )
    if corners is None:
        return []
    return [
        [float(x) / w, float(y) / h]
        for [x, y] in corners.reshape(-1, 2)
    ]


# ===== Helper: frame processing =====
def process_frame(frame: np.ndarray) -> dict:
    original_h, original_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    input_tensor = preprocess_image(pil_image)

    with torch.inference_mode():
        output = model(input_tensor)
        pred_mask = torch.argmax(output["out"], dim=1).squeeze().cpu().numpy()

    binary_mask = (pred_mask == 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"contour": [], "corners": [], "original_size": [original_w, original_h]}

    main_contour = max(contours, key=cv2.contourArea)

    # Scale contour from 512x512 back to original resolution
    scale_x = original_w / 512
    scale_y = original_h / 512
    main_contour_scaled = main_contour.astype(np.float32)
    main_contour_scaled[:, :, 0] *= scale_x
    main_contour_scaled[:, :, 1] *= scale_y
    main_contour_scaled = main_contour_scaled.astype(np.int32)

    contour_points = [
        [float(x) / original_w, float(y) / original_h]
        for [x, y] in main_contour_scaled.reshape(-1, 2)
    ]
    corners = detect_real_corners(main_contour_scaled, original_w, original_h)

    return {
        "contour": contour_points,
        "corners": corners,
        "original_size": [original_w, original_h],
    }


# ===== Endpoints =====
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    try:
        contents = await file.read()

        # Write to a temp file — cv2.VideoCapture requires a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
        finally:
            os.unlink(tmp_path)  # Always clean up the temp file

        if not ret:
            raise HTTPException(status_code=400, detail="Could not read a frame from the video.")

        result = process_frame(frame)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===== Entry point (local run) =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", port=port, reload=False)
