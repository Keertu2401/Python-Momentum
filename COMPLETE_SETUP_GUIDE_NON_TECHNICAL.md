# Complete Setup Guide - Non-Technical User

This guide explains exactly how to get your momentum strategy working with n8n, step by step, without assuming technical knowledge.

## 🤔 Understanding the Current Situation

Right now you have:
- ✅ **Code in GitHub** - Your momentum strategy code is stored in a GitHub repository
- ❌ **No Running API** - The code is not actually running anywhere that n8n can access
- ❌ **No Connection** - n8n cannot connect to your strategy because it's not running

Think of it like having a recipe (GitHub code) but no kitchen (running server) to cook the food.

## 🎯 What We Need to Accomplish

1. **Deploy the code** - Make your GitHub code run on a server
2. **Get a web address** - So n8n can call your strategy API
3. **Connect n8n** - Set up n8n to call your running strategy
4. **Test everything** - Make sure it all works together

## 📋 Step 1: Deploy Your Strategy (Choose ONE Option)

### Option A: Railway (Recommended - Easiest)

**What is Railway?** A service that takes your GitHub code and runs it on the internet automatically.

**Steps:**
1. **Go to Railway:**
   - Visit: https://railway.app
   - Click "Start a New Project"
   - Click "Login with GitHub"

2. **Connect Your Repository:**
   - Click "Deploy from GitHub repo"
   - Find your momentum strategy repository
   - Click "Deploy"

3. **Configure Environment:**
   - Railway will ask about the programming language → Select "Python"
   - It will automatically detect your requirements.txt file
   - Click "Deploy"

4. **Get Your URL:**
   - After deployment (5-10 minutes), you'll get a URL like:
   - `https://your-app-name.railway.app`
   - **SAVE THIS URL** - you'll need it for n8n

5. **Test Your Deployment:**
   - Open browser and go to: `https://your-app-name.railway.app/health`
   - You should see: `{"status": "healthy", "timestamp": "..."}`

### Option B: Render (Alternative)

**Steps:**
1. **Go to Render:**
   - Visit: https://render.com
   - Click "Get Started" → "GitHub"

2. **Create Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your momentum strategy repo

3. **Configuration:**
   - Name: `momentum-strategy-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`

4. **Deploy and Get URL:**
   - Click "Create Web Service"
   - Wait for deployment
   - You'll get a URL like: `https://momentum-strategy-api.onrender.com`

### Option C: Heroku (If you prefer)

**Steps:**
1. **Go to Heroku:**
   - Visit: https://heroku.com
   - Create account and login

2. **Create New App:**
   - Click "New" → "Create new app"
   - App name: `momentum-strategy-api`
   - Choose region closest to you

3. **Connect GitHub:**
   - Go to "Deploy" tab
   - Select "GitHub" as deployment method
   - Connect your repository

4. **Deploy:**
   - Click "Enable Automatic Deploys"
   - Click "Deploy Branch"
   - Get your URL: `https://momentum-strategy-api.herokuapp.com`

## 📊 Step 2: Verify Your API is Working

**Test Your Deployed API:**
1. **Health Check:**
   - Go to: `https://YOUR-APP-URL/health`
   - Should show: `{"status": "healthy"}`

2. **Strategy Test:**
   - Go to: `https://YOUR-APP-URL/run?capital=100000&top_n=5`
   - Should show momentum strategy results with stocks

**If you see errors:**
- Check the deployment logs in Railway/Render/Heroku
- Make sure all files (requirements.txt, config.yaml) are in your GitHub repo
- Contact me with the specific error message

## 🔧 Step 3: Update n8n Configuration

### Why I Removed Google Sheets from Clean Test

I removed Google Sheets because:
1. **Credentials are complex** - Google OAuth setup causes most import errors
2. **Start simple** - Get basic API connection working first
3. **Add features gradually** - Once basic works, add Google Sheets step by step

### Update Your n8n Workflow

**In your HTTP Request node:**
1. **Click on "API Call" node**
2. **Change the URL from:**
   ```
   http://localhost:8000/run
   ```
   **To your deployed URL:**
   ```
   https://YOUR-APP-URL/run
   ```

**Example:**
```
https://momentum-strategy-api.railway.app/run
```

## 📈 Step 4: Add Google Sheets Back (Step by Step)

Now let's add Google Sheets logging back to your workflow:

### 4.1 Create Google Sheets Credential

**Google Cloud Console Setup:**
1. **Go to:** https://console.cloud.google.com
2. **Create Project:**
   - Click "Select a project" → "New Project"
   - Name: "N8N Momentum Strategy"
   - Click "Create"

3. **Enable Google Sheets API:**
   - In the search bar, type "Google Sheets API"
   - Click on it → Click "Enable"

4. **Create Credentials:**
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: "Web application"
   - Name: "N8N Integration"
   - Authorized redirect URIs: Add your n8n callback URL
   - If n8n cloud: `https://app.n8n.cloud/rest/oauth2-credential/callback`
   - If self-hosted: `https://your-n8n-domain/rest/oauth2-credential/callback`

5. **Save Credentials:**
   - Copy the "Client ID" and "Client secret"
   - You'll need these for n8n

### 4.2 Create N8N Credential

**In N8N:**
1. **Go to:** Settings → Credentials
2. **Add Credential:** "Google Sheets OAuth2 API"
3. **Fill in:**
   - Credential Name: "Google Sheets - Momentum"
   - Client ID: [Paste from Google Cloud Console]
   - Client Secret: [Paste from Google Cloud Console]
4. **Click:** "Connect my account"
5. **Authorize:** Follow the Google authorization flow
6. **Verify:** Green checkmark appears

### 4.3 Prepare Google Sheet

**Create Sheet:**
1. **Go to:** https://sheets.google.com
2. **Create:** New blank spreadsheet
3. **Name it:** "Momentum Strategy Results"
4. **Rename Sheet1 to:** "Weekly_Reports"

**Add Headers (Row 1):**
```
A1: Date
B1: Capital
C1: Stocks
D1: Utilization
E1: Score
F1: Stock1
G1: Stock2
H1: Stock3
```

**Get Sheet ID:**
- From URL: `https://docs.google.com/spreadsheets/d/COPY_THIS_PART/edit`
- Copy the long ID between `/d/` and `/edit`

### 4.4 Add Google Sheets Node to Workflow

**In your n8n workflow:**
1. **Click the "+" after "Check Success" node**
2. **Search:** "Google Sheets"
3. **Select:** "Google Sheets"
4. **Configuration:**
   ```
   Credential: Google Sheets - Momentum
   Operation: Append or Update
   Document ID: [Paste your Sheet ID]
   Sheet Name: Weekly_Reports
   ```

5. **Column Mapping (Click "Add Column" for each):**
   ```
   Column: Date → Value: ={{ $json.date }}
   Column: Capital → Value: ={{ $json.capital }}
   Column: Stocks → Value: ={{ $json.stocks }}
   Column: Utilization → Value: ={{ $json.utilization }}
   Column: Score → Value: ={{ $json.score }}
   Column: Stock1 → Value: ={{ $json.stock1 }}
   Column: Stock2 → Value: ={{ $json.stock2 }}
   Column: Stock3 → Value: ={{ $json.stock3 }}
   ```

6. **Connect:** "Check Success" TRUE output to Google Sheets node

## 🧪 Step 5: Test Everything

### Test 1: Basic API Connection
1. **In n8n:** Click on "API Call" node
2. **Click:** "Execute Node"
3. **Expected:** Should show momentum strategy data
4. **If error:** Check your deployed URL is correct

### Test 2: Data Processing
1. **Click:** "Process Data" node
2. **Click:** "Execute Node" 
3. **Expected:** Should show clean, formatted data

### Test 3: Google Sheets
1. **Click:** "Google Sheets" node
2. **Click:** "Execute Node"
3. **Check Google Sheet:** New row should appear with data

### Test 4: Full Workflow
1. **Click:** "Execute Workflow" (top button)
2. **Watch:** All nodes should show green checkmarks
3. **Verify:** Data appears in Google Sheets

## 📱 Step 6: Add Slack Notifications (Optional)

### 6.1 Create Slack App

**Slack Setup:**
1. **Go to:** https://api.slack.com/apps
2. **Click:** "Create New App" → "From scratch"
3. **App Name:** "Momentum Strategy Bot"
4. **Workspace:** Select your Slack workspace

**Configure Permissions:**
1. **OAuth & Permissions** → Scopes
2. **Add Bot Token Scopes:**
   - `chat:write`
   - `channels:read`
3. **Install to Workspace**
4. **Copy:** "Bot User OAuth Token" (starts with xoxb-)

### 6.2 Create N8N Slack Credential

**In N8N:**
1. **Add Credential:** "Slack OAuth2 API"
2. **Access Token:** Paste your Bot User OAuth Token
3. **Test:** Should show green checkmark

### 6.3 Add Slack Node

**In Workflow:**
1. **Add Node:** "Slack" after Google Sheets
2. **Configuration:**
   ```
   Credential: Your Slack credential
   Channel: #general (or create #trading channel)
   Text:
   📊 Momentum Strategy Report
   Date: {{ $json.date }}
   Capital: ₹{{ ($json.capital / 100000).toFixed(1) }}L
   Stocks: {{ $json.stocks }}
   Utilization: {{ $json.utilization }}%
   Top Holdings: {{ $json.stock1 }}, {{ $json.stock2 }}
   ```

**Invite Bot to Channel:**
- In Slack: `/invite @Momentum Strategy Bot`

## ⏰ Step 7: Set Up Automation

### Schedule Configuration

**In your "Weekly Schedule" node:**
1. **Click on the Cron trigger node**
2. **Set schedule:**
   ```
   For weekly (Mondays 9 AM): 0 9 * * 1
   For daily (weekdays 9 AM): 0 9 * * 1-5
   For testing (every 5 minutes): */5 * * * *
   ```

### Enable Workflow
1. **Save workflow:** Ctrl+S
2. **Enable workflow:** Toggle switch at top right
3. **Verify:** Should show "Active"

## 🔍 Step 8: Monitoring and Maintenance

### Daily Checks
- **Check execution logs** in n8n
- **Verify data** in Google Sheets
- **Monitor Slack** notifications

### Weekly Checks
- **Review strategy performance**
- **Check API uptime**
- **Verify credentials haven't expired**

### Monthly Checks
- **Export workflow backup**
- **Review and optimize settings**
- **Update documentation**

## 🚨 Common Issues and Solutions

### "API Connection Failed"
**Problem:** n8n can't reach your API
**Solution:** 
1. Check your deployed URL is correct
2. Verify API is running (visit /health endpoint)
3. Check if URL needs https:// prefix

### "Google Sheets Permission Denied"
**Problem:** Credential issues
**Solution:**
1. Re-create Google Cloud credentials
2. Re-authenticate in n8n
3. Check sheet sharing permissions

### "No Data in Sheets"
**Problem:** Data not logging
**Solution:**
1. Check column names match exactly
2. Verify sheet name is correct
3. Test Google Sheets node individually

### "Slack Not Working"
**Problem:** Messages not appearing
**Solution:**
1. Invite bot to channel
2. Check bot token is correct
3. Verify channel name includes #

## 🎉 Success Checklist

Your setup is complete when:
- [ ] API is deployed and accessible via URL
- [ ] N8N can call your API successfully
- [ ] Data processes correctly
- [ ] Google Sheets logs data automatically
- [ ] Slack sends notifications (if configured)
- [ ] Workflow runs on schedule
- [ ] All tests pass

## 📞 Next Steps

Once everything is working:
1. **Monitor for 1 week** to ensure stability
2. **Add more features** like email notifications
3. **Integrate with broker APIs** for actual trading
4. **Set up alerts** for system failures
5. **Create backup procedures**

## 🆘 Getting Help

If you get stuck:
1. **Copy the exact error message**
2. **Note which step you're on**
3. **Take screenshots of your configuration**
4. **Share your deployed API URL** (so I can test it)

This approach ensures you have a working system without needing to understand the technical details!