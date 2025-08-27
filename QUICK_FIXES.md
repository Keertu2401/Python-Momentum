# Quick Fixes for Deployment Issues

## 🚨 Immediate Actions Required

### Fix 1: Update Your main.py File

**Problem:** Your main.py probably doesn't have the right code for deployment platforms.

**Solution:** Add this to the **END** of your main.py file:

```python
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (required for Railway/Render)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server with deployment-friendly settings
    uvicorn.run(
        "main:app",  # Reference to your FastAPI app
        host="0.0.0.0",  # Listen on all interfaces
        port=port,  # Use environment port
        reload=False,  # Disable reload in production
        access_log=True,  # Enable access logging
        log_level="info"  # Set appropriate log level
    )
```

**Replace any existing `if __name__ == "__main__":` block with the above.**

### Fix 2: Create Required Files in Your GitHub Repository

**Add these files to your repository root:**

#### File 1: `Procfile` (for Railway)
Create a new file named exactly `Procfile` (no extension):
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### File 2: `railway.toml` (for Railway backup)
Create a new file named `railway.toml`:
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"

[variables]
PORT = "8000"
ENVIRONMENT = "production"
```

#### File 3: `.python-version` (for version detection)
Create a new file named `.python-version`:
```
3.11.0
```

## 🔧 Render Configuration - Exact Fields

**When deploying on Render, use these EXACT values:**

### Basic Info:
```
Name: momentum-strategy-api
Environment: Python 3
Region: Oregon (US West)
Branch: main
```

### Build & Deploy:
```
Root Directory: 
(Leave this field EMPTY)

Build Command: 
pip install -r requirements.txt

Start Command: 
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables (Click "Advanced"):
```
Key: PORT
Value: 10000

Key: PYTHON_VERSION  
Value: 3.11.0

Key: ENVIRONMENT
Value: production
```

## 🚂 Railway Configuration

### If Railway Still Says "No Deployment":

**Option 1: Manual Service Creation**
1. Go to Railway dashboard
2. Click "New Project"
3. Choose "Empty Project"
4. Click "New Service" → "GitHub Repo"
5. Select your repository
6. Railway should now detect Python

**Option 2: Force Python Detection**
1. Ensure your repository has `requirements.txt` in the root
2. Add the files mentioned in Fix 2 above
3. Push changes to GitHub
4. Try deployment again

## 📂 Repository Structure Check

Your GitHub repository should look EXACTLY like this:
```
your-repository-name/
├── main.py              ← Your main API file
├── requirements.txt     ← Python dependencies
├── config.yaml         ← Configuration file
├── utils.py            ← Utility functions (if you have it)
├── Procfile            ← Railway deployment config
├── railway.toml        ← Railway backup config
├── .python-version     ← Python version
└── README.md           ← Documentation (optional)
```

**❌ Files should NOT be in subfolders like:**
- `/momentum-strategy/main.py`
- `/src/main.py`
- `/api/main.py`

## 🧪 Test Before Deploying

**Before trying deployment again:**

1. **Check your requirements.txt** - Should contain:
```
fastapi==0.110.0
uvicorn==0.29.0
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.40
PyYAML==6.0.2
python-dateutil==2.9.0.post0
```

2. **Verify main.py** - Should have the FastAPI app and the deployment code above

3. **Test locally** (if possible):
```bash
pip install -r requirements.txt
python main.py
```

## 🔄 Retry Steps

### For Railway:
1. **Add all the files** mentioned in Fix 2
2. **Commit and push** to GitHub
3. **Delete failed deployment** in Railway (if any)
4. **Create new project** → "Deploy from GitHub repo"
5. **Select repository** → Should now detect Python

### For Render:
1. **Update main.py** with Fix 1
2. **Add required files** to repository
3. **Push to GitHub**
4. **Create new web service** in Render
5. **Use EXACT configuration** from this guide

## 🆘 If Still Having Issues

**Send me these details:**

1. **Screenshot of your GitHub repository** (showing all files in root)
2. **Exact error message** from Railway or Render
3. **Your GitHub repository URL** (if it's public)
4. **Which deployment platform** you prefer to use

**Common Issues:**

- **Files in wrong location** → Move to repository root
- **Wrong start command** → Use exact command from this guide
- **Missing files** → Add Procfile and railway.toml
- **Wrong main.py ending** → Update with Fix 1 code

Follow these fixes step by step, and your deployment should work!