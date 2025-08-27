# N8N Quick Start Checklist

Use this checklist to set up your momentum strategy workflow step by step, avoiding common errors.

## ✅ Pre-Setup Checklist

### API Verification
- [ ] Momentum strategy API is running
- [ ] Test: `curl http://localhost:8000/health` returns success
- [ ] Test: `curl "http://localhost:8000/run?capital=10000&top_n=3"` returns data

### N8N Access
- [ ] N8N is installed and accessible
- [ ] You can create new workflows
- [ ] Version is n8n 1.0+ (check Help → About)

## 🔑 Step 1: Create Credentials (CRITICAL - Do This First!)

### Google Sheets Credential
1. **Google Cloud Console Setup:**
   - [ ] Go to [console.cloud.google.com](https://console.cloud.google.com)
   - [ ] Create project: "n8n-momentum"
   - [ ] Enable Google Sheets API
   - [ ] Create OAuth 2.0 credentials
   - [ ] Add redirect URI: `https://your-n8n-domain/rest/oauth2-credential/callback`

2. **N8N Credential Creation:**
   - [ ] Go to N8N → Settings → Credentials
   - [ ] Add "Google Sheets OAuth2 API"
   - [ ] Name: "Google Sheets - Trading"
   - [ ] Enter Client ID and Secret from Google
   - [ ] Click "Connect my account" and authorize
   - [ ] Verify green checkmark appears

### Slack Credential (Optional)
1. **Slack App Setup:**
   - [ ] Go to [api.slack.com](https://api.slack.com)
   - [ ] Create app → From scratch
   - [ ] Add to workspace
   - [ ] OAuth & Permissions → Add scopes: `chat:write`, `channels:read`
   - [ ] Install to workspace
   - [ ] Copy "Bot User OAuth Token"

2. **N8N Credential:**
   - [ ] Add "Slack OAuth2 API"
   - [ ] Name: "Slack - Trading"
   - [ ] Paste Bot Token
   - [ ] Test connection

## 📊 Step 2: Prepare Google Sheet

### Create Sheet
- [ ] Go to [sheets.google.com](https://sheets.google.com)
- [ ] Create new sheet: "Momentum Strategy Tracking"
- [ ] Rename first sheet to: "Weekly_Reports"

### Add Headers (Exact Names Required)
```
A1: Date
B1: Total_Capital  
C1: Selected_Stocks
D1: Utilization_Pct
E1: Avg_Momentum_Score
F1: Top_Stock_1
G1: Top_Stock_2
H1: Top_Stock_3
```

### Get Sheet ID
- [ ] Copy from URL: `https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit`
- [ ] Save this ID for later use

## 🔧 Step 3: Import Workflow

### Method A: Clean Import (Recommended)
- [ ] Download `n8n_clean_tested.json` file
- [ ] In N8N: Click "Import from File"
- [ ] Select the downloaded file
- [ ] Verify 4 nodes appear: Schedule → API Call → Process Data → Check Success

### Method B: Manual Creation (If Import Fails)
Follow the detailed manual steps in `N8N_VISUAL_SETUP_GUIDE.md`

## ⚙️ Step 4: Configure Workflow

### Update API URL
- [ ] Click on "API Call" node
- [ ] Update URL if needed:
  - Local: `http://localhost:8000/run`
  - Remote: `http://YOUR_SERVER_IP:8000/run`

### Test Basic Workflow
- [ ] Click "Execute Workflow"
- [ ] Verify all 4 nodes show green checkmarks
- [ ] Check "Process Data" node output shows proper data structure

## 📈 Step 5: Add Google Sheets Integration

### Add Google Sheets Node
- [ ] Click "+" after "Check Success" node
- [ ] Search "Google Sheets" → Select
- [ ] Choose "Google Sheets - Trading" credential
- [ ] Configuration:
  ```
  Operation: Append or Update
  Document ID: [Your Sheet ID]
  Sheet Name: Weekly_Reports
  ```

### Column Mapping
Click "Add Column" for each:
```
Column: Date          → Value: ={{ $json.date }}
Column: Total_Capital → Value: ={{ $json.capital }}
Column: Selected_Stocks → Value: ={{ $json.stocks }}
Column: Utilization_Pct → Value: ={{ $json.utilization }}
Column: Avg_Momentum_Score → Value: ={{ $json.score }}
Column: Top_Stock_1 → Value: ={{ $json.stock1 }}
Column: Top_Stock_2 → Value: ={{ $json.stock2 }}
Column: Top_Stock_3 → Value: ={{ $json.stock3 }}
```

### Connect Nodes
- [ ] Connect "Check Success" TRUE output to Google Sheets node

### Test Google Sheets
- [ ] Execute workflow
- [ ] Check Google Sheet for new row with data

## 📱 Step 6: Add Slack Notification (Optional)

### Add Slack Node
- [ ] Click "+" after Google Sheets
- [ ] Search "Slack" → Select
- [ ] Choose "Slack - Trading" credential
- [ ] Configuration:
  ```
  Channel: #trading
  Text: 📊 Momentum Strategy Report
  Date: {{ $json.date }}
  Capital: ₹{{ ($json.capital / 100000).toFixed(1) }}L
  Stocks: {{ $json.stocks }}
  Utilization: {{ $json.utilization }}%
  Score: {{ $json.score }}
  Top: {{ $json.stock1 }}, {{ $json.stock2 }}, {{ $json.stock3 }}
  ```

### Invite Bot to Channel
- [ ] In Slack, go to #trading channel
- [ ] Type: `/invite @your-bot-name`

### Test Slack
- [ ] Execute workflow
- [ ] Verify message appears in Slack channel

## 🧪 Step 7: Testing & Validation

### Complete Workflow Test
- [ ] Execute entire workflow manually
- [ ] All nodes show green checkmarks
- [ ] Data appears in Google Sheets
- [ ] Slack notification received (if configured)
- [ ] No error messages

### Schedule Test
- [ ] Set schedule to run in 5 minutes: `*/5 * * * *`
- [ ] Wait for automatic execution
- [ ] Verify scheduled run works
- [ ] Change back to weekly: `0 9 * * 1`

### Error Handling Test
- [ ] Stop API server temporarily
- [ ] Run workflow
- [ ] Verify graceful error handling
- [ ] Restart API and test again

## 🔒 Step 8: Production Setup

### Security
- [ ] Set proper API authentication (if required)
- [ ] Use environment variables for sensitive data
- [ ] Restrict Google Sheet access as needed

### Monitoring
- [ ] Save workflow: Ctrl+S
- [ ] Enable workflow: Toggle switch at top
- [ ] Set up execution history retention
- [ ] Configure error notifications

### Backup
- [ ] Export workflow JSON: Settings → Export
- [ ] Document all credential settings
- [ ] Save Google Sheet ID and configuration

## 🚨 Common Issues Quick Fix

### "Could not find credential"
- [ ] Create credentials BEFORE importing workflow
- [ ] Re-select credentials in each node

### "Google Sheets permission denied"  
- [ ] Re-authenticate Google credential
- [ ] Check OAuth scopes include Sheets access
- [ ] Verify sheet sharing permissions

### "API connection refused"
- [ ] Verify API is running: `curl http://localhost:8000/health`
- [ ] Check firewall/network settings
- [ ] Use correct IP address (not localhost if remote)

### "Slack channel not found"
- [ ] Include # in channel name: `#trading`
- [ ] Invite bot to channel: `/invite @bot-name`
- [ ] Check bot permissions

### "Invalid expression"
- [ ] Use exact syntax: `={{ $json.fieldName }}`
- [ ] Copy provided expressions exactly
- [ ] Check for typos in field names

## ✅ Success Criteria

Your setup is complete when:
- [ ] Workflow executes without errors
- [ ] Data logs to Google Sheets correctly
- [ ] Slack notifications work (if configured)  
- [ ] Schedule runs automatically
- [ ] All tests pass

## 📞 Next Steps

### Once Basic Workflow Works:
1. **Add email notifications**
2. **Implement broker API integration**
3. **Add portfolio comparison logic**
4. **Set up advanced monitoring**
5. **Configure rebalancing automation**

### For Advanced Features:
- Review `N8N_DETAILED_CONFIG_GUIDE.md`
- Use `n8n_complete_workflow.json` for full automation
- Implement order execution logic

## 🔄 Maintenance

### Weekly:
- [ ] Check execution logs
- [ ] Verify data quality
- [ ] Monitor API performance

### Monthly:
- [ ] Review credential expiry
- [ ] Update workflow if needed
- [ ] Backup configurations

This checklist should get you from zero to a working momentum strategy automation in under 30 minutes!