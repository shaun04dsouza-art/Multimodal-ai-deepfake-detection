# Multimodal AI Deepfake Detection System

A full-stack system for detecting AI-generated / manipulated media across **images, video, and text**. Built as a rapid single-day project, with a FastAPI backend, a browser-based dashboard, and an image classifier trained separately for future integration.

## Demo

The dashboard lets you drag-and-drop an image, video, or text file and get a real/fake prediction with a confidence score.

## Features

- **Image analysis** — Error Level Analysis (ELA), Laplacian noise variance, and color histogram entropy, combined into a forensic score.
- **Video analysis** — Samples frames at intervals and runs the same forensic pipeline, averaging results across frames.
- **Text analysis** — Heuristic scoring based on linguistic markers commonly found in AI-generated text.
- **Web dashboard** — Tabbed interface (Image / Video / Text) with drag-and-drop upload, live backend health check, and result visualization.
- **Image classifier (training pipeline)** — A ResNet18 CNN trained on a labeled real/fake image dataset with a leakage-free train/validation split (`train_image_model.py`), as a next step toward replacing the heuristic scorer with a learned model.

## Architecture

```
frontend/               → deepfake_app.html (dashboard UI, vanilla JS + Tailwind)
backend/                → main.py (FastAPI server)
train_image_model.py    → ResNet18 training script for image classification
START_SYSTEM.bat        → one-click launcher (starts backend + frontend, opens browser)
```

**Flow:** Frontend (port 3000) → FastAPI backend (port 8000) → forensic feature extraction → real/fake prediction with confidence.

## Tech Stack

- **Backend:** Python, FastAPI, OpenCV, Pillow, NumPy
- **ML/Training:** PyTorch, TorchVision (ResNet18), Scikit-learn (train/val split)
- **Frontend:** HTML, JavaScript, Tailwind CSS
- **Model (planned integration):** RoBERTa-based AI-text detector

## Current Limitations / Roadmap

This was built under tight time constraints (single day, ahead of a submission deadline), so a few things are intentionally simple right now:

- [ ] Wire the trained ResNet18 model into `/predict/image` and `/predict/video` (currently these use forensic heuristics, not the CNN)
- [ ] Replace the text keyword-heuristic with the downloaded RoBERTa AI-text-detector model
- [ ] Calibrate forensic score thresholds against a labeled validation set
- [ ] Add unified multimodal fusion layer combining image/video/text signals

## Setup

```bash
git clone https://github.com/<yourusername>/deepfake-detection-system.git
cd deepfake-detection-system
pip install -r requirements.txt
python main.py
```

Then open `frontend/deepfake_app.html` in your browser (or run `python -m http.server 3000` in the frontend folder).

On Windows, `START_SYSTEM.bat` automates all of this.

## Author

**Shaun Dsouza**
[LinkedIn](https://www.linkedin.com/in/shaun-dsouza-3680a92a2/) | [GitHub](https://github.com/shaun04dsouza-art)
