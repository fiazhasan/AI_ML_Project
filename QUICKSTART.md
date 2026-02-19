# Quick Start Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, CPU works but slower)
- 8GB+ RAM
- ~5GB disk space for dataset

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ML_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: Manual Download (Recommended)

1. Download Stanford Dogs dataset from: http://vision.stanford.edu/aditya86/ImageNetDogs/
2. Extract to `data/raw/` directory
3. Expected structure:
   ```
   data/raw/
   └── Images/
       ├── n02085620-Chihuahua/
       ├── n02085782-Japanese_spaniel/
       └── ...
   ```

### Option 2: Using Script

```bash
python scripts/download_data.py --data-dir data/raw
```

Note: The script provides instructions for manual download.

## Training

### Train Baseline Model

```bash
python scripts/train.py --model baseline --config config.yaml
```

### Train EfficientNet Model

```bash
python scripts/train.py --model efficientnet --config config.yaml
```

Training will:
- Automatically split data into train/val/test
- Apply data augmentation
- Handle class imbalance
- Save best model checkpoints
- Log metrics to TensorBoard

## Evaluation

```bash
python scripts/evaluate.py --model-path models/efficientnet/best_model.pth --config config.yaml
```

This will:
- Evaluate on test set
- Calculate metrics (accuracy, F1, etc.)
- Generate confusion matrix
- Perform error analysis
- Save visualizations

## Inference

### Command Line

```bash
python scripts/test_inference.py --image path/to/image.jpg --model-path models/efficientnet/best_model.pth
```

### API Server

```bash
# Start API server
python -m src.api.main

# Or using Docker
docker-compose up
```

### API Usage

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Open http://localhost:6006 in browser.

## Configuration

Edit `config.yaml` to customize:
- Data paths and splits
- Model hyperparameters
- Training settings
- Augmentation strategies
- API settings

## Project Structure

```
ML_project/
├── config.yaml          # Main configuration
├── requirements.txt     # Dependencies
├── README.md            # Full documentation
├── QUICKSTART.md        # This file
├── src/                 # Source code
├── scripts/             # Training/evaluation scripts
├── data/                # Dataset (not in repo)
├── models/              # Trained models (not in repo)
└── logs/                # Logs and visualizations
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in `config.yaml`
- Use gradient accumulation
- Enable mixed precision training

### Dataset Not Found

- Ensure dataset is in `data/raw/Images/`
- Check directory structure matches expected format

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.9+)

## Next Steps

1. Review `README.md` for detailed documentation
2. Explore `notebooks/` for interactive analysis
3. Check `logs/` for training visualizations
4. Customize `config.yaml` for your needs

## Support

For issues or questions, refer to:
- README.md for detailed documentation
- Code comments for implementation details
- Config file for all settings
