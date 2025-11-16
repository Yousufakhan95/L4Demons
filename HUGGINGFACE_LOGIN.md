# 🔐 HuggingFace Login & Setup

## 📋 **Quick Steps**

1. Get HuggingFace token
2. Set up for Modal (required)
3. Set up locally (optional)
4. Test connection

---

## 🎫 **Step 1: Get HuggingFace Token**

### **1.1 Create HuggingFace Account** (if you don't have one)

Visit: https://huggingface.co/join

### **1.2 Get Access Token**

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `L4Demons` (or any name you like)
4. Type: **Write** (needed for uploading models)
5. Click **"Generate token"**
6. **Copy the token** (looks like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

⚠️ **IMPORTANT:** Save this token securely! You won't be able to see it again.

---

## 🚀 **Step 2: Set Up for Modal** (Required)

### **2.1 Create Modal Secret**

```bash
# Replace hf_xxxxx with your actual token
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Example:**
```bash
modal secret create huggingface-secret HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

**Expected output:**
```
✓ Created secret huggingface-secret
```

### **2.2 Verify Secret**

```bash
modal secret list
```

**Should see:**
```
huggingface-secret
```

### **2.3 Test Modal Training**

```bash
modal run train_modal.py
```

**Should see:**
```
[HF] Using HuggingFace Hub for model storage
[HF] Repository: yousufakhan/L4Demons
```

---

## 💻 **Step 3: Set Up Locally** (Optional)

Only needed if you want to train locally or test locally.

### **Option A: Using CLI** (Recommended)

```bash
pip install huggingface-hub
huggingface-cli login
```

**Prompt:**
```
Token: [paste your token here]
Add token as git credential? (Y/n) y
```

### **Option B: Using Environment Variable**

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Linux/Mac:**
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### **Option C: Using Python**

```python
from huggingface_hub import login

login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

---

## 📁 **Step 4: Create Repository**

### **4.1 Create Model Repository**

1. Go to: https://huggingface.co/new
2. **Owner:** `yousufakhan` (your username)
3. **Model name:** `L4Demons`
4. **License:** MIT (or your preference)
5. **Make it public** ✓ (recommended)
6. Click **"Create model"**

**Your repo URL:** https://huggingface.co/yousufakhan/L4Demons

### **4.2 Verify Repository Exists**

Visit: https://huggingface.co/yousufakhan/L4Demons

Should see an empty repository (this is correct!)

---

## ✅ **Step 5: Test Connection**

### **Test Upload (Local)**

```python
from huggingface_hub import HfApi
import os

# Your token
token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Test upload
api = HfApi(token=token)

# Create a test file
with open("test.txt", "w") as f:
    f.write("Testing HuggingFace upload")

# Try uploading
try:
    api.upload_file(
        path_or_fileobj="test.txt",
        path_in_repo="test.txt",
        repo_id="yousufakhan/L4Demons",
        repo_type="model",
        commit_message="Test upload"
    )
    print("✅ Upload successful!")
    print("View at: https://huggingface.co/yousufakhan/L4Demons")
except Exception as e:
    print(f"❌ Upload failed: {e}")

# Clean up
os.remove("test.txt")
```

### **Test Modal Training**

```bash
modal run train_modal.py
```

**Look for these messages:**
```
[HF] Using HuggingFace Hub for model storage
[HF] Repository: yousufakhan/L4Demons
[ML] ✅ Initial weights uploaded successfully!
[ML]    View at: https://huggingface.co/yousufakhan/L4Demons
```

---

## 🐛 **Troubleshooting**

### **Issue: "Invalid token"**

**Causes:**
- Token copied incorrectly
- Token doesn't start with `hf_`
- Token expired

**Solution:**
1. Generate new token: https://huggingface.co/settings/tokens
2. Make sure it's type **"Write"**
3. Copy carefully (no extra spaces!)
4. Re-create Modal secret

---

### **Issue: "Repository not found"**

**Causes:**
- Repository doesn't exist
- Wrong username
- Wrong repo name

**Solution:**
1. Check repo exists: https://huggingface.co/yousufakhan/L4Demons
2. Verify username is `yousufakhan`
3. Verify repo name is `L4Demons` (case-sensitive!)
4. Make sure repo is public or you have access

---

### **Issue: "Permission denied"**

**Causes:**
- Token is "Read" type instead of "Write"
- Token doesn't have repo access

**Solution:**
1. Create new token with **"Write"** permissions
2. Update Modal secret:
```bash
modal secret update huggingface-secret HF_TOKEN=hf_newtoken
```

---

### **Issue: Modal can't find secret**

**Check secret exists:**
```bash
modal secret list
```

**If not listed, create it:**
```bash
modal secret create huggingface-secret HF_TOKEN=hf_yourtoken
```

**If exists but not working, recreate:**
```bash
modal secret delete huggingface-secret
modal secret create huggingface-secret HF_TOKEN=hf_yourtoken
```

---

## 📊 **Verification Checklist**

Before running training, verify:

- ✅ HuggingFace account created
- ✅ Access token generated (**Write** permission)
- ✅ Modal secret created (`huggingface-secret`)
- ✅ Repository created (`yousufakhan/L4Demons`)
- ✅ Repository is public (or you have access)
- ✅ Token starts with `hf_`

---

## 🎯 **Quick Reference**

### **Get Token**
https://huggingface.co/settings/tokens

### **Create Modal Secret**
```bash
modal secret create huggingface-secret HF_TOKEN=hf_yourtoken
```

### **Create Repository**
https://huggingface.co/new

### **Your Repository**
https://huggingface.co/yousufakhan/L4Demons

---

## 🚀 **Next Steps**

Once logged in:

1. **Test locally** (optional):
```bash
python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.whoami()
```

2. **Run Modal training**:
```bash
modal run train_modal.py
```

3. **Check HuggingFace**:
Visit https://huggingface.co/yousufakhan/L4Demons to see uploaded weights!

---

## 📚 **Related Documentation**

- **`HUGGINGFACE_SETUP.md`** - Detailed setup guide
- **`HUGGINGFACE_TRAINING_FLOW.md`** - How weights are managed
- **`TRAINING_PROGRESSION.md`** - Training lifecycle

---

**You're all set! Ready to train! 🚀🧠♟️**

