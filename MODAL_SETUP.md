# Modal Integration Guide for L4Demons

## Overview

This project is now integrated with **Modal** to run training on GPU infrastructure in the cloud.

## Prerequisites

1. **Install Modal CLI**:
   ```powershell
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```powershell
   modal token new
   ```
   This will open a browser window to authenticate. Save your token when prompted.

3. **Create a Modal Volume** (for persistent storage):
   ```powershell
   modal volume create l4demons-training-vol
   ```

## Running Training on Modal

### Basic Usage

Run training with default parameters:
```powershell
modal run train_modal.py
```

### Custom Parameters

Run with custom configuration:
```powershell
modal run train_modal.py --stockfish-games 50 --steps-per-batch 200 --batch-size 1024
```

### Available Parameters

- `--dataset-folder`: Path to datasets (default: `datasets`)
- `--max-positions`: Max positions to load (default: `200000`)
- `--stockfish-games`: Number of Stockfish games (default: `20`)
- `--steps-per-batch`: Training steps per batch (default: `100`)
- `--batch-size`: Batch size for training (default: `512`)
- `--learning-rate`: Learning rate (default: `1e-3`)

### Examples

```powershell
# Run a quick test with 5 games
modal run train_modal.py --stockfish-games 5

# Run with larger batch for faster training
modal run train_modal.py --batch-size 2048 --steps-per-batch 500

# Full training run
modal run train_modal.py --stockfish-games 100 --max-positions 500000
```

## What Happens

1. **Local**: Your machine launches a Modal app and calls the remote training function.
2. **Remote (Modal GPU)**:
   - A Docker container with PyTorch and dependencies is spun up
   - Your training code runs on GPU (A40 by default, configurable)
   - Models and data are persisted to `l4demons-training-vol`
   - Training results are streamed back to your terminal

3. **Output**: Training metrics (Elo, loss, accuracy) are printed in real-time to your console.

## GPU Options

To use a different GPU, edit `train_modal.py` and change this line in the `@app.function()` decorator:
```python
gpu="A40",  # Options: "T4", "L4", "A40", "A100", "H100", etc.
```

## Local Development

To still run training locally (without Modal):
```powershell
python -m src.main
```

## Troubleshooting

1. **"Token not found"**: Run `modal token new` to authenticate.
2. **"Volume not found"**: Run `modal volume create l4demons-training-vol`.
3. **"GPU not available"**: Check your Modal plan or try a different GPU type.
4. **"Module not found"**: Ensure `src/main.py` exists and is properly configured.

## File Structure

```
L4Demons/
├── src/
│   ├── main.py          (Training & inference logic)
│   └── utils/
├── train_modal.py       (Modal training entrypoint) ← NEW
├── requirements.txt     (Updated with modal)
├── model.pt             (Latest model)
├── model_best.pt        (Best model)
└── datasets/            (Training data)
```

## Next Steps

1. Install Modal: `pip install modal`
2. Authenticate: `modal token new`
3. Create volume: `modal volume create l4demons-training-vol`
4. Start training: `modal run train_modal.py --stockfish-games 20`
5. Monitor real-time Elo updates and training metrics in the terminal
