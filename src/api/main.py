"""
FastAPI application for inference – deployment-ready interface.
Input → preprocessing → model → output; config-driven; production-aware.
"""

from pathlib import Path
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
from typing import List

from src.api.schemas import (
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, PredictionItem
)
from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.models.model_factory import create_model
from src.inference.predictor import Predictor

# Setup logging
logger = setup_logger()

# Load configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="Dog Breed Classification API",
    description="ML Engineer Assessment – Dog Breed Classification Inference API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Predictor = None

# Max upload size from config (bytes)
MAX_FILE_SIZE = config.get("api.max_file_size", 10 * 1024 * 1024)  # default 10MB


def _get_class_names_from_checkpoint(model_path: str, num_classes: int) -> List[str]:
    """Load class names from checkpoint if saved (e.g. Colab best_model.pth)."""
    path = Path(model_path)
    if not path.exists():
        return [f"Breed_{i}" for i in range(num_classes)]
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        class_to_idx = ckpt.get("class_to_idx")
        if class_to_idx and isinstance(class_to_idx, dict):
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return [idx_to_class.get(i, f"Breed_{i}") for i in range(len(idx_to_class))]
    except Exception as e:
        logger.warning(f"Could not load class_to_idx from checkpoint: {e}")
    return [f"Breed_{i}" for i in range(num_classes)]


def load_model():
    """Load model at startup – config-driven, with real class names from checkpoint."""
    global predictor

    try:
        logger.info("Loading model...")
        model_path = config.get("inference.model_path")
        if not model_path or not Path(model_path).exists():
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        num_classes = config.get("data.num_classes", 120)
        class_names = _get_class_names_from_checkpoint(model_path, num_classes)

        model = create_model(
            "efficientnet",
            num_classes=num_classes,
            config=config["model"]
        )

        predictor = Predictor(
            model=model,
            model_path=model_path,
            class_names=class_names,
            device=config.get("training.device"),
            top_k=config.get("inference.top_k", 5)
        )

        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="EfficientNet-B0",
        num_classes=len(predictor.class_names),
        input_size=224
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict dog breed from single image.
    Input → preprocessing → model → output.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. image/jpeg, image/png)")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)} MB"
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Invalid image: {e}")
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")

    try:
        predictions = predictor.predict(image)
        prediction_items = [PredictionItem(**pred) for pred in predictions]
        return PredictionResponse(
            predictions=prediction_items,
            top_prediction=prediction_items[0]
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict dog breeds from multiple images (max 10, each under max file size)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    images = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            continue
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(img)
        except Exception:
            continue

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    try:
        batch_predictions = predictor.predict_batch(images)
        results = []
        for predictions in batch_predictions:
            items = [PredictionItem(**p) for p in predictions]
            results.append(PredictionResponse(predictions=items, top_prediction=items[0]))
        return BatchPredictionResponse(results=results)
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----- Demo UI: upload image, call /predict, show top-k breed predictions -----
DEMO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dog Breed Classifier | ML Inference Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f0f12;
      --surface: #1a1a1f;
      --surface2: #25252c;
      --border: #2e2e36;
      --text: #f4f4f5;
      --text-muted: #a1a1aa;
      --accent: #22c55e;
      --accent-dim: #16a34a;
      --danger: #ef4444;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
    }
    .app {
      max-width: 520px;
      margin: 0 auto;
      padding: 2rem 1.25rem;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    header h1 {
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 0.35rem;
    }
    header p {
      color: var(--text-muted);
      font-size: 0.9rem;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
      margin-bottom: 1.25rem;
      box-shadow: var(--shadow);
    }
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: var(--radius-sm);
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }
    .upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: rgba(34,197,94,0.06); }
    .upload-zone input[type="file"] { display: none; }
    .upload-zone .label { font-size: 0.95rem; color: var(--text-muted); margin-bottom: 0.75rem; display: block; }
    .upload-zone .hint { font-size: 0.8rem; color: var(--text-muted); opacity: 0.8; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: var(--radius-sm);
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: 1rem;
      transition: background 0.2s, transform 0.05s;
    }
    .btn:hover:not(:disabled) { background: var(--accent-dim); }
    .btn:active:not(:disabled) { transform: scale(0.98); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-upload { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
    .btn-upload:hover:not(:disabled) { background: var(--border); }
    .predict-row { margin-top: 1rem; text-align: center; }
    .upload-again { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); text-align: center; }
    .preview-wrap {
      text-align: center;
      margin-bottom: 1rem;
    }
    .preview-wrap img {
      max-width: 100%;
      max-height: 280px;
      border-radius: var(--radius-sm);
      object-fit: contain;
      border: 1px solid var(--border);
    }
    .result-card h3 {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
    }
    .top-pred {
      background: var(--surface2);
      border-radius: var(--radius-sm);
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
      border-left: 3px solid var(--accent);
    }
    .top-pred .name { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }
    .bar-wrap { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .bar { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.4s ease; }
    .pred-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.6rem 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }
    .pred-row:last-child { border-bottom: none; }
    .pred-row .name { color: var(--text-muted); }
    .pred-row .pct { font-weight: 500; color: var(--text); min-width: 3rem; text-align: right; }
    .loading { color: var(--text-muted); display: flex; align-items: center; gap: 0.5rem; }
    .loading::before {
      content: '';
      width: 18px; height: 18px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .error { color: var(--danger); font-size: 0.9rem; }
    .hidden { display: none !important; }
    .format-name { text-transform: capitalize; }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>Dog Breed Classifier</h1>
      <p>Upload an image to get top‑5 breed predictions</p>
    </header>

    <div class="card">
      <div class="upload-zone" id="zone">
        <input type="file" id="file" accept="image/*">
        <span class="label" id="fileLabel">No image selected</span>
        <span class="hint">JPG, PNG — max 10 MB. Drag and drop or click the button below.</span>
        <button type="button" class="btn btn-upload" id="uploadBtn">Upload Image</button>
      </div>
      <div class="predict-row hidden" id="predictRow">
        <button type="button" class="btn" id="btn">Predict breed</button>
      </div>
    </div>

    <div class="card preview-wrap hidden" id="previewCard">
      <img id="preview" alt="Preview">
    </div>

    <div class="card result-card hidden" id="resultCard">
      <h3>Predictions</h3>
      <div id="result"></div>
      <div class="upload-again">
        <button type="button" class="btn btn-upload" id="uploadAgainBtn">Upload Image</button>
      </div>
    </div>
  </div>

  <script>
    function formatBreed(s) {
      if (!s) return s;
      return s.replace(/^n\\d+-/, '').replace(/_/g, ' ').trim();
    }
    var zone = document.getElementById('zone');
    var fileInput = document.getElementById('file');
    var fileLabel = document.getElementById('fileLabel');
    var uploadBtn = document.getElementById('uploadBtn');
    var uploadAgainBtn = document.getElementById('uploadAgainBtn');
    var predictRow = document.getElementById('predictRow');
    var btn = document.getElementById('btn');
    var preview = document.getElementById('preview');
    var previewCard = document.getElementById('previewCard');
    var resultCard = document.getElementById('resultCard');
    var resultDiv = document.getElementById('result');

    function openFilePicker() { fileInput.click(); }
    uploadBtn.addEventListener('click', function(e) { e.stopPropagation(); openFilePicker(); });
    uploadAgainBtn.addEventListener('click', function(e) { e.stopPropagation(); openFilePicker(); });
    zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', function() { zone.classList.remove('dragover'); });
    zone.addEventListener('drop', function(e) {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
    });

    fileInput.addEventListener('change', function() {
      var has = this.files && this.files.length;
      btn.disabled = !has;
      resultCard.classList.add('hidden');
      if (has) {
        fileLabel.textContent = this.files[0].name;
        preview.src = URL.createObjectURL(this.files[0]);
        previewCard.classList.remove('hidden');
        predictRow.classList.remove('hidden');
      } else {
        fileLabel.textContent = 'No image selected';
        previewCard.classList.add('hidden');
        predictRow.classList.add('hidden');
      }
    });

    btn.addEventListener('click', async function(e) {
      e.stopPropagation();
      var file = fileInput.files[0];
      if (!file) return;
      btn.disabled = true;
      resultCard.classList.remove('hidden');
      resultDiv.innerHTML = '<div class="loading">Running inference…</div>';
      var form = new FormData();
      form.append('file', file);
      try {
        var r = await fetch('/predict', { method: 'POST', body: form });
        var data = await r.json();
        if (!r.ok) {
          resultDiv.innerHTML = '<p class="error">' + (data.detail || r.statusText) + '</p>';
          btn.disabled = false;
          return;
        }
        var top = data.top_prediction;
        var name = top.class || top.class_name || '';
        var pct = (100 * (top.confidence || 0)).toFixed(1);
        var html = '<div class="top-pred"><div class="name">' + formatBreed(name) + '</div><div class="bar-wrap"><div class="bar" style="width:' + pct + '%"></div></div><div style="margin-top:0.35rem;font-size:0.85rem;color:var(--text-muted)">' + pct + '% confidence</div></div>';
        (data.predictions || []).forEach(function(p, i) {
          var n = p.class || p.class_name || '';
          var c = (100 * (p.confidence || 0)).toFixed(1);
          html += '<div class="pred-row"><span class="name">' + (i+1) + '. ' + formatBreed(n) + '</span><span class="pct">' + c + '%</span></div>';
        });
        resultDiv.innerHTML = html;
      } catch (e) {
        resultDiv.innerHTML = '<p class="error">Request failed: ' + e.message + '</p>';
      }
      btn.disabled = false;
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def demo_ui():
    """Simple demo UI: upload image and see prediction (assessment / walkthrough)."""
    return HTMLResponse(DEMO_HTML)


if __name__ == "__main__":
    import uvicorn
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    uvicorn.run(app, host=host, port=port)
