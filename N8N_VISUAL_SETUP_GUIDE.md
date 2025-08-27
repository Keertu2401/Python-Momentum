# Visual N8N Setup Guide with Screenshots

This guide provides visual references for each configuration step to eliminate errors.

## 🖼️ Step-by-Step Visual Configuration

### Step 1: Creating Credentials

#### Google Sheets OAuth2 API Credential
1. **Navigate to Credentials**
   ```
   N8N Dashboard → Settings (gear icon) → Credentials
   ```

2. **Add New Credential**
   ```
   Click "Add Credential" button (top right)
   Search for "Google Sheets"
   Select "Google Sheets OAuth2 API"
   ```

3. **Credential Configuration Form**
   ```
   Field 1: Credential Name
   Value: "Google Sheets - Momentum Strategy"
   
   Field 2: Client ID
   Value: [From Google Cloud Console - OAuth 2.0 Client IDs]
   
   Field 3: Client Secret
   Value: [From Google Cloud Console - OAuth 2.0 Client IDs]
   ```

4. **OAuth Flow**
   ```
   Click "Connect my account" → Opens Google authorization
   Allow permissions → Returns to n8n with green checkmark
   Click "Save" button
   ```

#### Slack OAuth2 API Credential
1. **Add Slack Credential**
   ```
   Add Credential → Search "Slack" → Select "Slack OAuth2 API"
   ```

2. **Configuration**
   ```
   Credential Name: "Slack - Trading Alerts"
   Access Token: xoxb-your-bot-token-here
   ```

### Step 2: Creating Google Sheet Template

#### Sheet Creation
1. **Create New Google Sheet**
   ```
   Go to sheets.google.com
   Click "+" to create new sheet
   Name: "Momentum Strategy Tracking"
   ```

2. **Sheet 1 Setup: "Weekly_Reports"**
   ```
   A1: Date          B1: Total_Capital    C1: Selected_Stocks
   D1: Utilization_Pct  E1: Avg_Momentum_Score  F1: Top_Stock_1
   G1: Top_Stock_2   H1: Top_Stock_3
   ```

3. **Get Sheet ID**
   ```
   From URL: https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit
   Copy the SHEET_ID part (long string between /d/ and /edit)
   ```

### Step 3: Manual Node Creation (Recommended)

#### Node 1: Schedule Trigger
1. **Add Node**
   ```
   Click "+" in workflow canvas
   Search "Cron" → Select "Cron"
   ```

2. **Node Configuration**
   ```
   Mode: Custom
   Expression: 0 9 * * 1
   ```
   **Visual:** You should see "Every Monday at 09:00"

#### Node 2: HTTP Request
1. **Add Node**
   ```
   Click "+" → Search "HTTP Request" → Select
   ```

2. **Basic Configuration**
   ```
   Method: GET (dropdown)
   URL: http://localhost:8000/run
   ```

3. **Query Parameters Section**
   ```
   Click "Add Parameter" button 3 times
   
   Parameter 1:
     Name: capital
     Value: 100000
   
   Parameter 2:
     Name: top_n
     Value: 5
   
   Parameter 3:
     Name: price_cap
     Value: 3000
   ```

4. **Options Section**
   ```
   Timeout: 60000
   Response Format: JSON (should be default)
   ```

#### Node 3: Code Processor
1. **Add Node**
   ```
   Click "+" → Search "Code" → Select "Code"
   ```

2. **Code Configuration**
   ```javascript
   // Paste this exact code - no modifications needed
   const input = $input.all()[0].json;

   if (!input || !input.summary) {
     throw new Error('Invalid API response');
   }

   const summary = input.summary;
   const topStocks = input.top || [];

   const result = {
     executionDate: new Date().toISOString().split('T')[0],
     executionTime: new Date().toISOString(),
     totalCapital: summary.total_capital || 0,
     selectedStocks: summary.selected_stocks || 0,
     utilization: Math.round((summary.utilization_pct || 0) * 100) / 100,
     avgMomentumScore: Math.round((summary.avg_momentum_score || 0) * 100) / 100,
     topStock1: topStocks[0]?.Ticker || '',
     topStock2: topStocks[1]?.Ticker || '',
     topStock3: topStocks[2]?.Ticker || '',
     status: 'SUCCESS'
   };

   return { json: result };
   ```

### Step 4: Node Connections

#### Connecting Nodes
1. **Connect Cron to HTTP Request**
   ```
   Hover over Cron node → See small circle on right side
   Click and drag from circle to HTTP Request node
   ```

2. **Connect HTTP Request to Code**
   ```
   Same process: Drag from HTTP Request output to Code input
   ```

**Visual Check:** You should see arrows connecting the nodes in sequence.

### Step 5: Google Sheets Integration

#### Node 4: Google Sheets
1. **Add Node**
   ```
   Click "+" → Search "Google Sheets" → Select
   ```

2. **Authentication**
   ```
   Credential: Select "Google Sheets - Momentum Strategy"
   (Should show green checkmark if credential is working)
   ```

3. **Operation Configuration**
   ```
   Resource: Document (should be default)
   Operation: Append or Update
   ```

4. **Document Settings**
   ```
   Document ID: [Paste your Google Sheet ID here]
   Sheet Name: Weekly_Reports
   ```

5. **Column Mapping** (Click "Add Column" for each)
   ```
   Column 1:
     Column: Date
     Value: ={{ $json.executionDate }}
   
   Column 2:
     Column: Total_Capital
     Value: ={{ $json.totalCapital }}
   
   Column 3:
     Column: Selected_Stocks
     Value: ={{ $json.selectedStocks }}
   
   Column 4:
     Column: Utilization_Pct
     Value: ={{ $json.utilization }}
   
   Column 5:
     Column: Avg_Momentum_Score
     Value: ={{ $json.avgMomentumScore }}
   
   Column 6:
     Column: Top_Stock_1
     Value: ={{ $json.topStock1 }}
   
   Column 7:
     Column: Top_Stock_2
     Value: ={{ $json.topStock2 }}
   
   Column 8:
     Column: Top_Stock_3
     Value: ={{ $json.topStock3 }}
   ```

6. **Connect to Code Node**
   ```
   Drag from Code node output to Google Sheets input
   ```

### Step 6: Slack Notification

#### Node 5: Slack
1. **Add Node**
   ```
   Click "+" → Search "Slack" → Select
   ```

2. **Authentication**
   ```
   Credential: Select "Slack - Trading Alerts"
   ```

3. **Configuration**
   ```
   Resource: Message (default)
   Operation: Post Message
   Channel: #trading
   ```

4. **Message Content**
   ```
   Text: Copy this exactly:
   
   📊 *Momentum Strategy Weekly Report*

   📅 Date: {{ $json.executionDate }}
   💰 Capital: ₹{{ ($json.totalCapital / 100000).toFixed(1) }}L
   📈 Stocks Selected: {{ $json.selectedStocks }}
   ⚡ Utilization: {{ $json.utilization }}%
   🎯 Avg Score: {{ $json.avgMomentumScore }}

   🔝 Top Holdings:
   1. {{ $json.topStock1 }}
   2. {{ $json.topStock2 }}
   3. {{ $json.topStock3 }}
   ```

5. **Advanced Options**
   ```
   Markdown: Enable (toggle on)
   ```

## 🧪 Testing Each Step

### Test 1: HTTP Request Node
1. **Click on HTTP Request node**
2. **Click "Execute Node" button**
3. **Expected Result:**
   ```json
   {
     "summary": {
       "total_capital": 100000,
       "selected_stocks": 5,
       "utilization_pct": 85.2,
       "avg_momentum_score": 4.5
     },
     "top": [
       {
         "Ticker": "RELIANCE.NS",
         "Price": 1384.9,
         "MomentumScore": 5.2
       }
     ]
   }
   ```

### Test 2: Code Node
1. **Click on Code node**
2. **Click "Execute Node"**
3. **Expected Result:**
   ```json
   {
     "executionDate": "2024-01-15",
     "executionTime": "2024-01-15T09:00:00.000Z",
     "totalCapital": 100000,
     "selectedStocks": 5,
     "utilization": 85.2,
     "avgMomentumScore": 4.5,
     "topStock1": "RELIANCE.NS",
     "topStock2": "",
     "topStock3": "",
     "status": "SUCCESS"
   }
   ```

### Test 3: Google Sheets Node
1. **Click on Google Sheets node**
2. **Click "Execute Node"**
3. **Check Google Sheet:** New row should appear with data

### Test 4: Slack Node
1. **Click on Slack node**
2. **Click "Execute Node"**
3. **Check Slack channel:** Message should appear

### Test 5: Full Workflow
1. **Click "Execute Workflow"**
2. **Watch each node execute in sequence**
3. **Verify all outputs**

## 🚨 Troubleshooting Common Issues

### Issue 1: "Could not find credential"
**Visual Signs:**
- Red exclamation mark on node
- Error message in node execution

**Solution:**
1. Click on the node with error
2. Look for "Credential" dropdown
3. Select the correct credential from dropdown
4. Save workflow

### Issue 2: Google Sheets "Permission Denied"
**Visual Signs:**
- Google Sheets node shows error
- Error message mentions permissions

**Solution:**
1. Go back to Google Sheets credential
2. Click "Reconnect" or "Test Connection"
3. Re-authorize with Google
4. Ensure you have edit permissions on the sheet

### Issue 3: Slack "Channel Not Found"
**Visual Signs:**
- Slack node execution fails
- Error about channel not existing

**Solution:**
1. Verify channel name includes # (e.g., #trading)
2. Invite the Slack bot to the channel:
   ```
   In Slack: /invite @your-bot-name
   ```
3. Check bot has permission to post messages

### Issue 4: API Connection Failed
**Visual Signs:**
- HTTP Request node times out
- "Connection refused" or "404" errors

**Solution:**
1. Verify API is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. If using remote n8n, replace localhost with actual IP:
   ```
   http://YOUR_SERVER_IP:8000/run
   ```

### Issue 5: Expression Errors in Code Node
**Visual Signs:**
- Code node shows syntax errors
- Red error indicators

**Solution:**
1. Copy the exact code provided above
2. Don't modify variable names
3. Ensure proper JSON structure

## 📋 Configuration Checklist

### Before Starting
- [ ] API is running and accessible
- [ ] Google account has Sheets access
- [ ] Slack workspace is accessible
- [ ] N8N is properly installed

### Credentials Setup
- [ ] Google Sheets OAuth2 credential created and tested
- [ ] Slack OAuth2 credential created and tested
- [ ] All credentials show green checkmarks

### Google Sheet Preparation
- [ ] Sheet created with correct name
- [ ] Headers match exactly (case sensitive)
- [ ] Sheet ID copied correctly
- [ ] Sharing permissions set (if needed)

### Node Configuration
- [ ] Cron trigger set to correct schedule
- [ ] HTTP Request URL is correct
- [ ] Query parameters added correctly
- [ ] Code node has exact code provided
- [ ] Google Sheets node has correct document ID
- [ ] Slack node has correct channel name

### Testing
- [ ] Individual nodes execute successfully
- [ ] Full workflow executes without errors
- [ ] Data appears in Google Sheets
- [ ] Slack notifications are received
- [ ] API logs show requests

### Final Checks
- [ ] Workflow saved
- [ ] Schedule is active
- [ ] Error handling configured
- [ ] Monitoring set up

## 📧 Support & Next Steps

### If You're Still Having Issues:

1. **Export your workflow:**
   ```
   Settings → Export → Download workflow JSON
   ```

2. **Check error logs:**
   ```
   Click on failed node → Check execution data tab
   ```

3. **Test individual components:**
   ```
   Test API directly: curl http://localhost:8000/run
   Test credentials: Try simple operations
   ```

### Gradual Enhancement:
1. **Start with basic workflow** (3 nodes)
2. **Add Google Sheets** when basic works
3. **Add Slack** when logging works
4. **Add advanced features** gradually

This visual guide should resolve most configuration issues. The key is to test each component individually before connecting them together.