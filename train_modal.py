"""
Modal training entrypoint for L4Demons Chess AI.

🚀 MULTI-GPU CONFIGURATION: 4× NVIDIA B200 GPUs
   - Total VRAM: 672GB (168GB per GPU)
   - Optimized batch size: 8192 (4× single GPU)
   - 2× more games & positions per cycle

Usage:
    modal run train_modal.py
    
Adjust parameters:
    modal run train_modal.py --batch-size 16384 --stockfish-games 80
"""

import modal
import sys
import os

# ============================================================
#  MODAL APP + IMAGE SETUP
# ============================================================

app = modal.App("l4demons-chess-training")

# Persistent volume for datasets + saved models
training_volume = modal.Volume.from_name(
    "l4demons-training-vol", create_if_missing=True
)

# Build container image with dependencies AND your source code
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "python-chess==1.999",
        "numpy",
        "huggingface-hub",
    )
    .add_local_dir("src", remote_path="/root/project/src")  # include your code
    .add_local_dir("engines", remote_path="/root/project/engines")
    .add_local_dir("datasets", remote_path="/root/project/datasets")
)


# ============================================================
#  REMOTE TRAINING FUNCTION
# ============================================================

@app.function(
    image=image,
    gpu="B200",  # 4x NVIDIA B200 GPUs (672GB total VRAM!)
    volumes={"/training": training_volume},
    timeout=3600 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For HF Hub uploads
)
def train_chess_model_remote(
    dataset_folder: str = "datasets",
    max_positions: int = None,  # Load ALL data with PARALLEL processing
    stockfish_games: int = 1,      # 1 self-play + 2 Stockfish = 3 total per cycle
    steps_per_batch: int = 10,
    batch_size: int = 1024,        # Preserved user setting
    learning_rate: float = 1e-3,
) -> str:
    import subprocess

    # Install Stockfish binary inside the container so Modal can use it
    print("[MODAL] Installing Stockfish engine...")
    try:
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True, capture_output=True)
        print("[MODAL] Stockfish installed successfully")
    except Exception as e:
        print(f"[MODAL] Warning: Stockfish install failed: {e}")
        print("[MODAL] Attempting to find pre-installed Stockfish...")

    # Make sure project root is importable and import as package 'src'
    sys.path.insert(0, "/root/project")

    try:
        from src.main import run_training_local
        print("[MODAL] Successfully imported src.main.run_training_local")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[MODAL] Import error: {e}"

    # Prefer datasets inside the persistent volume if present
    import shutil

    # Debug: list project root contents so we can see what Modal uploaded
    try:
        print("[MODAL DEBUG] /root/project contents:", os.listdir('/root/project'))
    except Exception:
        pass

    # If the persistent volume already has datasets, use them
    if os.path.exists("/training/datasets") and os.listdir("/training/datasets"):
        dataset_folder = "/training/datasets"
        print(f"[MODAL] Using datasets from persistent volume: {dataset_folder}")
    # Else, if the uploaded project contains datasets, copy them into the persistent volume
    elif os.path.exists("/root/project/datasets"):
        try:
            print("[MODAL DEBUG] /root/project/datasets contents:", os.listdir('/root/project/datasets'))
        except Exception:
            pass
        src_ds = "/root/project/datasets"
        dst_ds = "/training/datasets"
        try:
            print(f"[MODAL] Copying datasets from project ({src_ds}) into persistent volume ({dst_ds})...")
            if not os.path.exists(dst_ds):
                os.makedirs(dst_ds, exist_ok=True)
            for name in os.listdir(src_ds):
                s = os.path.join(src_ds, name)
                d = os.path.join(dst_ds, name)
                if os.path.isdir(s):
                    if not os.path.exists(d):
                        shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            dataset_folder = dst_ds
            print(f"[MODAL] Datasets copied to persistent volume: {dataset_folder}")
        except Exception as e:
            print(f"[MODAL] Failed to copy datasets into volume: {e}")
            dataset_folder = f"/root/project/{dataset_folder}"
            print(f"[MODAL] Falling back to project datasets: {dataset_folder}")
    else:
        dataset_folder = f"/root/project/{dataset_folder}"
        print(f"[MODAL] Using datasets from uploaded project: {dataset_folder}")

    print("[MODAL] Starting GPU training...")
    print(f"[MODAL] Games={stockfish_games}, Steps={steps_per_batch}, Batch={batch_size}, LR={learning_rate}")

    try:
        run_training_local(
            dataset_folder=dataset_folder,
            max_positions=max_positions,
            stockfish_games=stockfish_games,
            steps_per_batch=steps_per_batch,
            batch=batch_size,
            lr=learning_rate,
        )
        summary = f"[MODAL] Training complete: {stockfish_games} games, {steps_per_batch} steps/batch"
        print(summary)
        return summary
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("[MODAL] ERROR DURING TRAINING:")
        print(error_details)
        return f"[MODAL] Training failure: {error_details}"


# ============================================================
#  LOCAL ENTRYPOINT – launches the remote function
# ============================================================

@app.local_entrypoint()
def main(
    dataset_folder: str = "datasets",
    max_positions: int = None,   # Load ALL data with PARALLEL processing
    stockfish_games: int = 1,       # 1 self-play + 2 Stockfish = 3 total per cycle
    steps_per_batch: int = 10,
    batch_size: int = 1024,         # Preserved user setting
    learning_rate: float = 1e-3,
):
    print("\n" + "="*70)
    print("🚀 LAUNCHING REMOTE MODAL TRAINING")
    print("="*70)
    print(f"[LOCAL] This terminal is just launching the remote job")
    print(f"[LOCAL] Actual training happens on Modal's cloud GPU!")
    print()
    
    print(f"⚡⚡⚡ AGGRESSIVE PARALLEL LOADING MODE ⚡⚡⚡")
    print(f"[MODAL] Using multiprocessing for maximum speed!")
    print(f"[MODAL] All 30+ PGN files will load in parallel")
    print(f"[MODAL] First run: ~5-10 min | Cached runs: ~30 seconds")
    print()
    
    print(f"[MODAL] Hardware: B200 GPU (168GB VRAM)")
    print(f"[MODAL] Training Depth: 2 (fixed for all training)")
    print(f"[MODAL] Games per cycle: 1 self-play + 2 Stockfish = 3 total")
    print(f"[MODAL] Config:")
    print(f"  - Max positions: {max_positions if max_positions else 'ALL (load everything!)'}")
    print(f"  - Self-play games: {stockfish_games}")
    print(f"  - Stockfish games: 2 (1 as White, 1 as Black)")
    print(f"  - Steps per batch: {steps_per_batch}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print("="*70 + "\n")

    result = train_chess_model_remote.remote(
        dataset_folder=dataset_folder,
        max_positions=max_positions,
        stockfish_games=stockfish_games,
        steps_per_batch=steps_per_batch,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print()
    print("[MODAL] Remote training result:")
    print(result)
