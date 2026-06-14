import os
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
MODEL_PATH = os.path.join(tempfile.gettempdir(), "Final_best_model.pth")
GDRIVE_FILE_ID = "1lZOWFBteOX9HnE4B82RMTbmLp46-mPhp"
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


# ===== Google Drive Downloader =====
def download_from_gdrive(file_id: str, dest_path: str):
    """
    Download a large file from Google Drive.
    Tries multiple URL strategies to handle Google's anti-scraping measures.
    """
    import re
    import requests

    logger.info(f"Downloading model from Google Drive (id={file_id})...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    def stream_to_disk(response) -> float:
        """Write streamed response to disk, return size in MB."""
        total_bytes = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
        return total_bytes / (1024 * 1024)

    def is_html(response) -> bool:
        ct = response.headers.get("Content-Type", "")
        return "text/html" in ct

    # ── Strategy 1: drive.google.com/uc with confirm=t (2024+ trick) ──
    logger.info("Trying strategy 1: uc?confirm=t ...")
    try:
        r = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": "t"},
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        if not is_html(r):
            size_mb = stream_to_disk(r)
            if size_mb >= 1.0:
                logger.info(f"Strategy 1 success: {size_mb:.1f} MB")
                return
    except Exception as e:
        logger.warning(f"Strategy 1 failed: {e}")

    # ── Strategy 2: drive.usercontent.google.com (newer endpoint) ──
    logger.info("Trying strategy 2: drive.usercontent.google.com ...")
    try:
        r = session.get(
            "https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        if not is_html(r):
            size_mb = stream_to_disk(r)
            if size_mb >= 1.0:
                logger.info(f"Strategy 2 success: {size_mb:.1f} MB")
                return
    except Exception as e:
        logger.warning(f"Strategy 2 failed: {e}")

    # ── Strategy 3: fetch page, extract real download URL from HTML ──
    logger.info("Trying strategy 3: parse HTML for download URL ...")
    try:
        page = session.get(
            f"https://drive.google.com/file/d/{file_id}/view",
            timeout=30,
        )
        # Find the direct download link embedded in the page
        match = re.search(
            r'https://drive\.usercontent\.google\.com/download[^"\'<>\s]+',
            page.text
        )
        if not match:
            match = re.search(
                r'"downloadUrl":"([^"]+)"',
                page.text
            )
        if match:
            dl_url = match.group(0).replace("\\u003d", "=").replace("\\u0026", "&")
            r = session.get(dl_url, stream=True, timeout=300)
            r.raise_for_status()
            if not is_html(r):
                size_mb = stream_to_disk(r)
                if size_mb >= 1.0:
                    logger.info(f"Strategy 3 success: {size_mb:.1f} MB")
                    return
    except Exception as e:
        logger.warning(f"Strategy 3 failed: {e}")

    # ── All strategies failed ──
    if os.path.exists(dest_path):
        os.remove(dest_path)
    raise RuntimeError(
        "All Google Drive download strategies failed.\n"
        "Please verify:\n"
        "  1. The file is shared as 'Anyone with the link can view'\n"
        "  2. The file ID is correct: " + file_id
    )


# ===== Model Loading (lifespan) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Download model if not already cached
    if not os.path.exists(MODEL_PATH):
        try:
            download_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    else:
        logger.info(f"Using cached model at {MODEL_PATH}")

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
        "model_path": MODEL_PATH,
        "model_cached": os.path.exists(MODEL_PATH),
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
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
