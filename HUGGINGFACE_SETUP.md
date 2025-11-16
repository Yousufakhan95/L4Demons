# Hugging Face Hub Integration Setup

This project is configured to automatically sync model weights with Hugging Face Hub at:
**https://huggingface.co/yousufakhan/L4Demons**

## Features

✅ **Automatic Model Downloads**: When the model starts, it tries to download the latest weights from HF Hub  
✅ **Automatic Uploads**: During training, improved models are automatically pushed to HF Hub  
✅ **Version Control**: Every training checkpoint is saved with descriptive commit messages

## Setup Instructions

### 1. Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "L4Demons Training")
4. Select "Write" permissions (needed for uploading models)
5. Copy the token (starts with `hf_...`)

### 2. For Local Training

Set the environment variable before running training:

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "hf_your_token_here"
python src/main.py
```

**Linux/Mac:**
```bash
export HF_TOKEN="hf_your_token_here"
python src/main.py
```

Or add it to your `.env` file permanently.

### 3. For Modal Training

Create a Modal secret with your HF token:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Then run your training as usual:
```bash
modal run train_modal.py
```

### 4. Verify Setup

When training starts, you should see:
- `[HF] Downloading model.pt from yousufakhan/L4Demons...` (on startup)
- `[HF] Uploading ... to yousufakhan/L4Demons/...` (during training)

## Model Files on HF Hub

The following files are synced:

- **`model.pt`**: Latest training checkpoint (updated every cycle)
- **`model_best.pt`**: Best model based on validation loss

## Troubleshooting

### "Failed to download" errors on startup
This is normal if the model hasn't been uploaded yet. The system will start training from scratch.

### "Failed to upload" errors during training
- Check that your HF token has write permissions
- Verify the token is correctly set in your environment
- Make sure the repository exists at https://huggingface.co/yousufakhan/L4Demons
- You may need to create the repository first on Hugging Face

### Repository doesn't exist
Create it at: https://huggingface.co/new

1. Repository name: `L4Demons`
2. Owner: `yousufakhan`
3. Type: Model
4. Visibility: Public or Private (your choice)

## Benefits

🚀 **Easy Deployment**: Anyone can load your latest model by just running the code  
💾 **Backup**: All model checkpoints are safely stored in the cloud  
🔄 **Version History**: Track model improvements over time  
📊 **Sharing**: Share your model with collaborators or the community  
🌐 **Multi-machine**: Train on Modal, serve from local machine, same weights!

