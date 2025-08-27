# Detailed Deployment Troubleshooting - Railway & Render

This guide addresses specific deployment issues you're encountering.

## 🚂 Railway Deployment Issues

### Issue: "No deployment for this service"

**Why this happens:**
1. **Missing required files** - Railway can't detect what type of app you have
2. **Wrong repository structure** - Files might be in a subfolder
3. **No Python detection** - Railway doesn't recognize it as a Python app
4. **Missing start configuration** - No way to run the app

### Solution Steps for Railway:

#### Step 1: Check Your Repository Structure
Your GitHub repository should look like this:
```
your-repo-name/
├── main.py                 ← Main API file
├── requirements.txt        ← Python dependencies  
├── config.yaml            ← Configuration file
├── utils.py               ← Utility functions (if exists)
├── test_api.py            ← Test file (if exists)
└── README.md              ← Documentation (optional)
```

**If files are in a subfolder:**
- Railway looks at the root directory
- If your Python files are in `/momentum-strategy/` or similar, move them to root
- Or use a different deployment approach

#### Step 2: Verify requirements.txt Content
Your `requirements.txt` should contain:
```
fastapi==0.110.0
uvicorn[standard]==0.29.0
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.40
PyYAML==6.0.2
python-dateutil==2.9.0.post0
requests==2.31.0
aiohttp==3.9.3
```

#### Step 3: Add Railway Configuration (Option 1)
Create a file called `railway.toml` in your repository root:
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
```

#### Step 4: Add Procfile (Option 2)
Create a file called `Procfile` (no extension) in your repository root:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### Step 5: Verify main.py Structure
Make sure your `main.py` ends with:
```python
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

#### Step 6: Railway Deployment Process
1. **Go to Railway Dashboard**
2. **Delete the failed service** (if exists)
3. **Create New Project** → "Deploy from GitHub repo"
4. **Select Repository** → Choose your momentum strategy repo
5. **Wait for Detection** → Should now show "Python" detected
6. **Check Build Logs** → Click on deployment to see what's happening

### Railway Environment Variables
After deployment, set these in Railway dashboard:
```
PORT=8000
ENVIRONMENT=production
```

## 🔧 Render Deployment - Detailed Configuration

### All Required Fields for Render:

#### Step 1: Basic Information
```
Name: momentum-strategy-api
Environment: Python 3
Region: Choose closest to your location (e.g., Oregon for US West)
Branch: main (or master)
```

#### Step 2: Build & Deploy Settings
```
Root Directory: . 
(Leave empty if files are in repository root)

Build Command: 
pip install -r requirements.txt

Start Command: 
uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### Step 3: Environment Variables
Click "Advanced" and add:
```
Key: PORT
Value: 10000

Key: ENVIRONMENT  
Value: production

Key: PYTHON_VERSION
Value: 3.11.0
```

#### Step 4: Instance Type
```
Free (for testing)
Starter ($7/month for production)
```

#### Step 5: Auto-Deploy
```
Enable: Yes (so it updates when you push to GitHub)
```

### Complete Render Configuration Screenshot Guide:

**Page 1: Repository Selection**
- Connect GitHub account
- Select your momentum strategy repository
- Click "Connect"

**Page 2: Service Configuration**
```
Service Name: momentum-strategy-api
Environment: Python 3
Region: Oregon (US West) or closest to you
Branch: main
Root Directory: [Leave empty]
```

**Page 3: Build Settings**
```
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Page 4: Environment Variables (Click Advanced)**
```
PORT = 10000
ENVIRONMENT = production
PYTHON_VERSION = 3.11.0
```

**Page 5: Billing**
```
Instance Type: Free
```

**Page 6: Review & Deploy**
- Review all settings
- Click "Create Web Service"

## 🔍 Troubleshooting Specific Errors

### Render Error: "Build failed"
**Check these:**
1. **requirements.txt exists** in repository root
2. **All dependencies are valid** - no typos in package names
3. **Python version compatibility** - use Python 3.9-3.11

**Fix:**
```
# If error mentions specific packages, update requirements.txt:
fastapi>=0.100.0,<1.0.0
uvicorn>=0.20.0,<1.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
yfinance>=0.2.0,<1.0.0
PyYAML>=6.0.0,<7.0.0
python-dateutil>=2.8.0,<3.0.0
```

### Render Error: "Start command failed"
**Common issues:**
1. **Wrong start command** - Using incorrect syntax
2. **Port not bound correctly** - App not listening on $PORT
3. **main.py issues** - File not found or syntax errors

**Fix:**
1. **Verify your main.py has this at the end:**
```python
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

2. **Use exact start command:**
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Railway Error: "Nixpacks build failed"
**This means Railway can't figure out how to build your app.**

**Fix:**
1. **Add explicit Python version file** - Create `.python-version`:
```
3.11.0
```

2. **Add railway.toml** configuration:
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"

[variables]
PORT = "8000"
ENVIRONMENT = "production"
```

## 📂 Repository Structure Verification

### Check Your GitHub Repository:

**Go to your GitHub repository and verify you see these files at the root level:**

```
✅ main.py (your API code)
✅ requirements.txt (dependencies)  
✅ config.yaml (configuration)
✅ utils.py (if you have it)
❌ Should NOT be in subfolders like /src/ or /momentum-strategy/
```

**If files are in subfolders:**
1. **Option A: Move files to root**
   - Download all Python files
   - Upload them to repository root
   - Delete the subfolder

2. **Option B: Update deployment settings**
   - In Render: Set "Root Directory" to your subfolder name
   - In Railway: This is more complex, better to move files

## 🧪 Testing Your Repository Structure

**Before deploying, verify locally:**

1. **Clone your repository:**
```bash
git clone https://github.com/your-username/your-repo-name
cd your-repo-name
```

2. **Check files exist:**
```bash
ls -la
# Should show: main.py, requirements.txt, config.yaml
```

3. **Test requirements.txt:**
```bash
pip install -r requirements.txt
# Should install without errors
```

4. **Test main.py:**
```bash
python main.py
# Should start the server
```

## 🔄 Step-by-Step Railway Retry

**After fixing repository structure:**

1. **Go to Railway dashboard**
2. **Delete failed deployment** (if any)
3. **Create new project**
4. **Deploy from GitHub repo**
5. **Select your repository**
6. **Wait for automatic detection**
7. **Should now show "Python" instead of unknown**
8. **Deployment should start automatically**

## 🔄 Step-by-Step Render Retry

**With correct configuration:**

1. **Go to Render dashboard**
2. **New → Web Service**
3. **Connect repository**
4. **Fill exact fields as shown above**
5. **Double-check start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Click "Create Web Service"**
7. **Watch build logs** for any errors

## 📋 Verification Checklist

Before deploying, check:
- [ ] main.py exists in repository root
- [ ] requirements.txt exists and has valid Python packages
- [ ] config.yaml exists (your strategy configuration)
- [ ] No files are in subfolders (unless using root directory setting)
- [ ] Repository is public or properly connected to deployment service
- [ ] All Python files have no syntax errors

## 🆘 If Still Having Issues

**For Railway:**
1. Check the build logs in Railway dashboard
2. Copy the exact error message
3. Verify your repository structure matches the examples above

**For Render:**
1. Watch the deployment logs
2. Note the exact error at which step (Build or Deploy)
3. Verify the start command is exactly: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Send me:**
1. **Screenshot of your GitHub repository** (showing file structure)
2. **Exact error message** from Railway or Render
3. **Your repository URL** (if public)
4. **Which step failed** (detection, build, or deploy)

This should resolve both Railway and Render deployment issues!