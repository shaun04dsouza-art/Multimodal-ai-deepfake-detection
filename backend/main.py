import os
import tempfile
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageChops, ImageEnhance, ImageStat


# ==========================================================
#   FASTAPI + CORS FIX
# ==========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # FRONTEND can connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
#   FORENSIC ANALYSIS HELPERS
# ==========================================================

# ---------- ELA ----------
def ela_score(img_path, quality=90):
    try:
        orig = Image.open(img_path).convert("RGB")

        # Temp compressed file
        temp_path = img_path + "_tmp.jpg"
        orig.save(temp_path, "JPEG", quality=quality)

        compressed = Image.open(temp_path)
        diff = ImageChops.difference(orig, compressed)
        diff = ImageEnhance.Brightness(diff).enhance(10)

        arr = np.array(diff).astype(np.float32)
        score = float(np.mean(arr))

        os.remove(temp_path)
        return score
    except:
        return 0.0


# ---------- Noise Variance ----------
def noise_score(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        noise = cv2.Laplacian(img, cv2.CV_64F).var()
        return float(noise)
    except:
        return 0.0


# ---------- Color Entropy ----------
def entropy_score(img_path):
    try:
        img = Image.open(img_path)
        stat = ImageStat.Stat(img)
        ent = 0
        for channel in stat.histogram():
            total = sum(channel)
            for count in channel:
                if count != 0:
                    p = count / total
                    ent -= p * np.log2(p)
        return float(ent)
    except:
        return 0.0


# ---------- Blend Prediction ----------
def blend_prediction(ela, noise, entropy):
    """
    Normalize & combine forensic features into final probability.
    """

    ela_norm = min(1.0, ela / 35)
    noise_norm = 1.0 - min(1.0, noise / 1500)
    ent_norm = 1.0 - min(1.0, entropy / 7)

    fake_score = (ela_norm * 0.5) + (noise_norm * 0.3) + (ent_norm * 0.2)
    fake_score = max(0, min(fake_score, 1))

    real_score = 1 - fake_score

    pred = "fake" if fake_score > real_score else "real"
    conf = max(fake_score, real_score) * 100

    return pred, conf, fake_score * 100, real_score * 100


# ==========================================================
#   HEALTH CHECK
# ==========================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu", "model": "multi-forensic-v2"}


# ==========================================================
#   IMAGE PREDICTION
# ==========================================================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save file
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            path = tmp.name

        # Scores
        ela = ela_score(path)
        noise = noise_score(path)
        ent = entropy_score(path)

        pred, conf, fake_p, real_p = blend_prediction(ela, noise, ent)

        os.remove(path)

        return {
            "prediction": pred,
            "confidence": f"{conf:.2f}%",
            "fake_probability": f"{fake_p:.2f}%",
            "real_probability": f"{real_p:.2f}%",
            "ela": ela,
            "noise": noise,
            "entropy": ent
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================================
#   VIDEO PREDICTION
# ==========================================================
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            video = tmp.name

        cap = cv2.VideoCapture(video)
        frame_id = 0

        ela_scores = []
        noise_scores = []
        ent_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % 15 != 0:
                continue

            # Save frame temp
            frame_path = video + "_frame.jpg"
            cv2.imwrite(frame_path, frame)

            ela_scores.append(ela_score(frame_path))
            noise_scores.append(noise_score(frame_path))
            ent_scores.append(entropy_score(frame_path))

            os.remove(frame_path)

        cap.release()
        os.remove(video)

        if len(ela_scores) == 0:
            return {"prediction": "error", "confidence": "0%"}

        ela = float(np.mean(ela_scores))
        noise = float(np.mean(noise_scores))
        ent = float(np.mean(ent_scores))

        pred, conf, fake_p, real_p = blend_prediction(ela, noise, ent)

        return {
            "prediction": pred,
            "confidence": f"{conf:.2f}%",
            "fake_probability": f"{fake_p:.2f}%",
            "real_probability": f"{real_p:.2f}%",
            "frames_analyzed": len(ela_scores),
            "ela": ela,
            "noise": noise,
            "entropy": ent
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================================
#   TEXT PREDICTION
# ==========================================================
@app.post("/predict/text")
async def predict_text(file: UploadFile = File(...)):
    try:
        text = (await file.read()).decode("utf-8", errors="ignore").lower()

        markers = ["as an ai", "i am an ai", "language model",
                   "i cannot", "chatgpt", "openai", "gpt-", "claude"]

        score = sum(20 for m in markers if m in text)
        score = min(score, 95)

        pred = "fake" if score > 50 else "real"
        conf = max(score, 100 - score)

        return {
            "prediction": pred,
            "confidence": f"{conf}%",
            "fake_probability": f"{score}%",
            "real_probability": f"{100 - score}%"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================================
#   RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
