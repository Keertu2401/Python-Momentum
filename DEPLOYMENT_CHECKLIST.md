# Simple Deployment Checklist

Follow this checklist step by step to deploy your momentum strategy from GitHub to a live API.

## ✅ Pre-Deployment Check

- [ ] Your momentum strategy code is in a GitHub repository
- [ ] Repository contains these files:
  - [ ] `main.py` (main API file)
  - [ ] `requirements.txt` (Python dependencies)
  - [ ] `config.yaml` (configuration file)
  - [ ] Any other Python files from the strategy

## 🚀 Option 1: Railway Deployment (Recommended)

### Step 1: Sign Up and Connect
- [ ] Go to https://railway.app
- [ ] Click "Start a New Project"
- [ ] Click "Login with GitHub" and authorize Railway

### Step 2: Deploy Repository
- [ ] Click "Deploy from GitHub repo"
- [ ] Select your momentum strategy repository
- [ ] Click "Deploy"

### Step 3: Configure
- [ ] Railway auto-detects Python
- [ ] Waits for build to complete (5-10 minutes)
- [ ] Note down your app URL (looks like: `https://momentum-strategy-production.railway.app`)

### Step 4: Test Deployment
- [ ] Visit: `https://your-app-url.railway.app/health`
- [ ] Should see: `{"status": "healthy", "timestamp": "..."}`
- [ ] Test strategy: `https://your-app-url.railway.app/run?capital=10000&top_n=3`

## 🚀 Option 2: Render Deployment (Alternative)

### Step 1: Sign Up
- [ ] Go to https://render.com
- [ ] Click "Get Started" → "GitHub"

### Step 2: Create Service
- [ ] Click "New +" → "Web Service"
- [ ] Connect GitHub account
- [ ] Select your momentum strategy repository

### Step 3: Configure
- [ ] Name: `momentum-strategy-api`
- [ ] Environment: `Python 3`
- [ ] Build Command: `pip install -r requirements.txt`
- [ ] Start Command: `python main.py`
- [ ] Instance Type: Free (for testing)

### Step 4: Deploy and Test
- [ ] Click "Create Web Service"
- [ ] Wait for deployment (10-15 minutes)
- [ ] Test your URL: `https://momentum-strategy-api.onrender.com/health`

## 📊 Google Sheets Setup

### Step 1: Create Sheet
- [ ] Go to https://sheets.google.com
- [ ] Create new spreadsheet
- [ ] Name: "Momentum Strategy Results"
- [ ] Rename "Sheet1" to "Weekly_Reports"

### Step 2: Add Headers (Row 1)
```
A1: Date
B1: Capital
C1: Stocks
D1: Utilization
E1: Score
F1: Stock1
G1: Stock2
H1: Stock3
I1: Price1
J1: Price2
K1: Price3
```

### Step 3: Get Sheet ID
- [ ] Copy from URL: `https://docs.google.com/spreadsheets/d/COPY_THIS_PART/edit`
- [ ] Save this ID for n8n configuration

## 🔑 Google Cloud Console Setup

### Step 1: Create Project
- [ ] Go to https://console.cloud.google.com
- [ ] Click "Select a project" → "New Project"
- [ ] Name: "N8N Momentum Strategy"
- [ ] Click "Create"

### Step 2: Enable API
- [ ] Search "Google Sheets API"
- [ ] Click it → Click "Enable"

### Step 3: Create OAuth Credentials
- [ ] Click "Create Credentials" → "OAuth client ID"
- [ ] Configure consent screen if prompted
- [ ] Application type: "Web application"
- [ ] Name: "N8N Integration"
- [ ] Authorized redirect URIs: 
  - For n8n cloud: `https://app.n8n.cloud/rest/oauth2-credential/callback`
  - For self-hosted: `https://your-n8n-domain/rest/oauth2-credential/callback`

### Step 4: Save Credentials
- [ ] Copy "Client ID"
- [ ] Copy "Client secret"
- [ ] Keep these safe for n8n setup

## 🔧 N8N Workflow Configuration

### Step 1: Create Google Sheets Credential
- [ ] In n8n: Settings → Credentials
- [ ] Add "Google Sheets OAuth2 API"
- [ ] Name: "Google Sheets - Momentum"
- [ ] Paste Client ID and Client Secret
- [ ] Click "Connect my account" and authorize

### Step 2: Import Workflow
- [ ] Use `n8n_complete_with_sheets.json` file
- [ ] Import into n8n

### Step 3: Update Configuration
- [ ] In "API Call" node, change URL to your deployed URL:
  ```
  https://your-app-url.railway.app/run
  ```
- [ ] In "Google Sheets" node, update Document ID with your Sheet ID
- [ ] Select your Google Sheets credential

### Step 4: Test
- [ ] Execute workflow manually
- [ ] Check Google Sheets for new data
- [ ] Verify all nodes show green checkmarks

## 📱 Slack Setup (Optional)

### Step 1: Create Slack App
- [ ] Go to https://api.slack.com/apps
- [ ] "Create New App" → "From scratch"
- [ ] Name: "Momentum Strategy Bot"
- [ ] Select your workspace

### Step 2: Configure Permissions
- [ ] OAuth & Permissions → Bot Token Scopes
- [ ] Add: `chat:write` and `channels:read`
- [ ] "Install to Workspace"
- [ ] Copy "Bot User OAuth Token"

### Step 3: N8N Slack Credential
- [ ] In n8n: Add "Slack OAuth2 API"
- [ ] Paste Bot User OAuth Token
- [ ] Test connection

### Step 4: Invite Bot
- [ ] In Slack: Go to #trading channel (or create it)
- [ ] Type: `/invite @Momentum Strategy Bot`

## ⏰ Schedule Setup

### Set Workflow Schedule
- [ ] In "Weekly Schedule" node
- [ ] For Monday 9 AM: `0 9 * * 1`
- [ ] For daily 9 AM: `0 9 * * 1-5`
- [ ] For testing (every 5 min): `*/5 * * * *`

### Enable Workflow
- [ ] Save workflow (Ctrl+S)
- [ ] Toggle "Active" switch
- [ ] Verify status shows "Active"

## 🧪 Final Testing

### Manual Test
- [ ] Click "Execute Workflow"
- [ ] All nodes show green checkmarks
- [ ] Data appears in Google Sheets
- [ ] Slack notification received (if configured)

### Automatic Test
- [ ] Set 5-minute schedule for testing
- [ ] Wait for automatic execution
- [ ] Verify it runs automatically
- [ ] Change back to weekly schedule

## 📋 Success Criteria

You're done when:
- [ ] ✅ API is deployed and accessible
- [ ] ✅ Health check returns success
- [ ] ✅ Strategy endpoint returns data
- [ ] ✅ N8N can call your API
- [ ] ✅ Google Sheets logs data automatically
- [ ] ✅ Workflow runs on schedule
- [ ] ✅ All notifications work

## 🆘 Common Issues

### API Won't Deploy
**Check:**
- [ ] All required files in GitHub repo
- [ ] requirements.txt exists and complete
- [ ] main.py exists
- [ ] No syntax errors in code

### API Deployed but Not Working
**Check:**
- [ ] Visit /health endpoint
- [ ] Check deployment logs
- [ ] Verify all dependencies installed

### N8N Can't Connect
**Check:**
- [ ] API URL is correct in n8n
- [ ] URL includes https://
- [ ] API is actually running

### Google Sheets Not Working
**Check:**
- [ ] Credential is authenticated
- [ ] Sheet ID is correct
- [ ] Column headers match exactly
- [ ] Sheet permissions allow editing

### No Data Appearing
**Check:**
- [ ] Workflow is active
- [ ] Schedule is correct
- [ ] No error messages in execution log
- [ ] API returns valid data

## 📞 Next Steps

Once everything works:
1. **Monitor daily** for first week
2. **Backup workflow** (export JSON)
3. **Document any customizations**
4. **Add email notifications** if needed
5. **Consider broker API integration** for actual trading

Your momentum strategy is now automated and running! 🎉