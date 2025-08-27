# Detailed N8N Configuration Guide - Step by Step

This guide addresses common import errors and provides exact configuration steps for setting up the momentum strategy in n8n.

## 🚨 Common Import Errors & Solutions

### Error 1: "Invalid workflow JSON format"
**Problem**: JSON syntax errors or invalid node structure
**Solution**: Use the corrected workflow files provided below

### Error 2: "Missing credentials type"
**Problem**: Workflow references credentials that don't exist
**Solution**: Create credentials first, then import workflow

### Error 3: "Node type not found"
**Problem**: Workflow uses nodes not available in your n8n version
**Solution**: Use compatible node versions (provided below)

### Error 4: "Connection errors between nodes"
**Problem**: Invalid node connections in workflow
**Solution**: Manually reconnect nodes after import

## 🔧 Step-by-Step Configuration

### Step 1: Create Credentials FIRST (Before Import)

#### 1.1 Google Sheets OAuth2 API Credential
1. **Go to N8N Settings** → Credentials → Add Credential
2. **Select**: "Google Sheets OAuth2 API"
3. **Configuration**:
   ```
   Name: Google Sheets - Momentum Strategy
   Client ID: [Get from Google Cloud Console]
   Client Secret: [Get from Google Cloud Console]
   ```

**Getting Google Credentials**:
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project or select existing
3. Enable Google Sheets API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URI: `https://your-n8n-domain/rest/oauth2-credential/callback`
6. Copy Client ID and Client Secret to n8n

#### 1.2 Slack OAuth2 API Credential  
1. **Add Credential**: "Slack OAuth2 API"
2. **Configuration**:
   ```
   Name: Slack - Trading Alerts
   Client ID: [From Slack App]
   Client Secret: [From Slack App]
   Access Token: [Bot User OAuth Token]
   ```

**Getting Slack Credentials**:
1. Go to [api.slack.com](https://api.slack.com)
2. Create new app → From scratch
3. Add to your workspace
4. Go to OAuth & Permissions
5. Add scopes: `chat:write`, `channels:read`
6. Install to workspace
7. Copy "Bot User OAuth Token"

#### 1.3 SMTP Email Credential
1. **Add Credential**: "SMTP"
2. **Configuration**:
   ```
   Name: Email - Trading Reports
   User: your-email@gmail.com
   Password: your-app-password
   Host: smtp.gmail.com
   Port: 587
   Secure: true
   ```

#### 1.4 HTTP Header Auth (for Broker API)
1. **Add Credential**: "Header Auth"
2. **Configuration**:
   ```
   Name: Zerodha Kite API
   Name: Authorization
   Value: token api_key:access_token
   ```

### Step 2: Create Google Sheet Template

Create a Google Sheet with these exact sheet names and headers:

#### Sheet 1: "Rebalancing_Log"
| Column A | Column B | Column C | Column D | Column E | Column F | Column G | Column H | Column I | Column J | Column K | Column L |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Timestamp | Date | Total_Orders | Buy_Orders | Sell_Orders | Total_Order_Value | Total_Capital | Utilization_Pct | Avg_Momentum_Score | Market_Open | Validated_Orders | Rejected_Orders |

#### Sheet 2: "Order_Execution_Log"
| Column A | Column B | Column C | Column D | Column E | Column F | Column G | Column H | Column I | Column J | Column K | Column L |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Timestamp | Date | Ticker | Action | Quantity | Order_Type | Status | Order_ID | Momentum_Score | Priority | Estimated_Value | Tag |

#### Sheet 3: "Weekly_Reports"
| Column A | Column B | Column C | Column D | Column E | Column F | Column G | Column H |
|----------|----------|----------|----------|----------|----------|----------|----------|
| Date | Total_Capital | Selected_Stocks | Utilization_Pct | Avg_Momentum_Score | Top_Stock_1 | Top_Stock_2 | Top_Stock_3 |

**Get Sheet ID**: Copy from URL between `/d/` and `/edit`
```
https://docs.google.com/spreadsheets/d/1a2b3c4d5e6f7g8h9i0j/edit
Sheet ID = 1a2b3c4d5e6f7g8h9i0j
```

### Step 3: Clean Workflow Import

Instead of the complex workflow, let's start with a simple, working version:

## 🎯 Method 1: Import Minimal Working Workflow

1. **Download**: Use `n8n_minimal_working.json` file
2. **Import Process**:
   - Go to n8n dashboard
   - Click "Import from File" or "Templates" → "Import"
   - Select the `n8n_minimal_working.json` file
   - Click "Import"

3. **Verify Import**:
   - You should see 3 nodes: Schedule Trigger → Get Strategy Data → Process Data
   - No errors should appear

## 🔨 Method 2: Manual Node Creation (If Import Fails)

If the import still fails, create nodes manually:

### Node 1: Cron Trigger
1. **Add Node** → Trigger → Cron
2. **Parameters**:
   ```
   Trigger Interval: Custom
   Expression: 0 9 * * 1
   ```
3. **Position**: (240, 300)

### Node 2: HTTP Request
1. **Add Node** → Regular → HTTP Request
2. **Parameters**:
   ```
   URL: http://localhost:8000/run
   Method: GET
   Query Parameters:
     - Name: capital, Value: 100000
     - Name: top_n, Value: 5
   ```
3. **Position**: (460, 300)

### Node 3: Code
1. **Add Node** → Data Transformation → Code
2. **Code**:
   ```javascript
   // Process the momentum strategy response
   const input = $input.all()[0].json;

   // Check if we have valid data
   if (!input || !input.summary) {
     throw new Error('Invalid API response');
   }

   const summary = input.summary;
   const topStocks = input.top || [];

   // Create a simple summary
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
3. **Position**: (680, 300)

### Node Connections
1. **Connect**: Cron → HTTP Request
2. **Connect**: HTTP Request → Code

## 📊 Step 4: Add Google Sheets Logging

### Node 4: Google Sheets
1. **Add Node** → Google → Google Sheets
2. **Credential**: Select your "Google Sheets - Momentum Strategy" credential
3. **Parameters**:
   ```
   Operation: Append or Update
   Document ID: YOUR_SHEET_ID_HERE
   Sheet Name: Weekly_Reports
   Columns:
     - Date: ={{ $json.executionDate }}
     - Total_Capital: ={{ $json.totalCapital }}
     - Selected_Stocks: ={{ $json.selectedStocks }}
     - Utilization_Pct: ={{ $json.utilization }}
     - Avg_Momentum_Score: ={{ $json.avgMomentumScore }}
     - Top_Stock_1: ={{ $json.topStock1 }}
     - Top_Stock_2: ={{ $json.topStock2 }}
     - Top_Stock_3: ={{ $json.topStock3 }}
   ```
4. **Connect**: Code → Google Sheets

## 📱 Step 5: Add Slack Notification

### Node 5: Slack
1. **Add Node** → Communication → Slack
2. **Credential**: Select your "Slack - Trading Alerts" credential
3. **Parameters**:
   ```
   Channel: #trading
   Text: 📊 Momentum Strategy Report
   Date: {{ $json.executionDate }}
   Capital: ₹{{ ($json.totalCapital / 100000).toFixed(1) }}L
   Stocks: {{ $json.selectedStocks }}
   Utilization: {{ $json.utilization }}%
   Top Holdings: {{ $json.topStock1 }}, {{ $json.topStock2 }}, {{ $json.topStock3 }}
   ```
4. **Connect**: Code → Slack

## 🔧 Detailed Configuration Steps

### Configuring HTTP Request Node

1. **Open HTTP Request Node**
2. **Basic Settings**:
   ```
   Method: GET
   URL: http://localhost:8000/run
   ```

3. **Query Parameters** (click Add Parameter for each):
   ```
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

4. **Options** → **Timeout**: 60000 (60 seconds)

### Configuring Google Sheets Node

1. **Operation**: "Append or Update"
2. **Document ID**: 
   - Click the field
   - Paste your Google Sheet ID (from URL)
   - Example: `1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t`

3. **Sheet Name**: `Weekly_Reports` (exact name from your sheet)

4. **Column Mapping** (click "Add Column" for each):
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

### Configuring Slack Node

1. **Channel**: 
   - Type: `#trading` (or your channel name)
   - Make sure the bot is invited to this channel

2. **Text**:
   ```
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

3. **Other Options** → **Markdown**: Enable

## 🧪 Step 6: Testing Configuration

### Test 1: Manual Execution
1. **Click on HTTP Request node**
2. **Click "Execute Node"**
3. **Check output**: Should show momentum strategy data
4. **Expected Output**:
   ```json
   {
     "summary": {
       "total_capital": 100000,
       "selected_stocks": 5,
       "utilization_pct": 85.2
     },
     "top": [
       {"Ticker": "STOCK1.NS", "Price": 1200},
       {"Ticker": "STOCK2.NS", "Price": 850}
     ]
   }
   ```

### Test 2: Full Workflow
1. **Click "Execute Workflow"** button
2. **Watch each node execute**
3. **Check outputs**:
   - Code node should process data correctly
   - Google Sheets should log the data
   - Slack should send notification

### Test 3: Verify Integrations
1. **Google Sheets**: Check if new row appeared
2. **Slack**: Verify message was posted
3. **API**: Check API logs for requests

## 🚨 Common Configuration Errors & Fixes

### Error: "Could not find credential"
**Solution**:
1. Go to Settings → Credentials
2. Create the missing credential type
3. Re-select credential in the node

### Error: "Google Sheets permission denied"
**Solution**:
1. Check OAuth scopes include Google Sheets
2. Re-authenticate Google credential
3. Verify sheet sharing permissions

### Error: "Slack channel not found"
**Solution**:
1. Verify channel name (include #)
2. Invite bot to the channel
3. Check bot permissions

### Error: "API connection refused"
**Solution**:
1. Verify API is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Use correct IP address (not localhost if n8n is remote)

### Error: "Invalid expression in Code node"
**Solution**:
1. Copy the exact code provided above
2. Check for typos in variable names
3. Verify JSON structure matches API response

## 📋 Step 7: Advanced Configuration

### Adding Email Notifications

1. **Add Node**: Email Send
2. **Credential**: Select SMTP credential
3. **Configuration**:
   ```
   From: trading@yourcompany.com
   To: portfolio@yourcompany.com
   Subject: Weekly Momentum Report - {{ $json.executionDate }}
   HTML Body:
   <h2>📊 Momentum Strategy Report</h2>
   <p><strong>Date:</strong> {{ $json.executionDate }}</p>
   <p><strong>Capital:</strong> ₹{{ ($json.totalCapital / 100000).toFixed(1) }}L</p>
   <p><strong>Selected Stocks:</strong> {{ $json.selectedStocks }}</p>
   <p><strong>Utilization:</strong> {{ $json.utilization }}%</p>
   <p><strong>Avg Score:</strong> {{ $json.avgMomentumScore }}</p>
   <h3>Top Holdings:</h3>
   <ul>
     <li>{{ $json.topStock1 }}</li>
     <li>{{ $json.topStock2 }}</li>
     <li>{{ $json.topStock3 }}</li>
   </ul>
   ```

### Adding Error Handling

1. **Add Node**: IF
2. **Configuration**:
   ```
   Condition: {{ $json.status }} equals SUCCESS
   ```
3. **Connect**: 
   - IF TRUE → Continue to notifications
   - IF FALSE → Error notification

### Adding Webhook Trigger

1. **Replace Cron with Webhook**
2. **Add Node**: Webhook
3. **Configuration**:
   ```
   HTTP Method: POST
   Path: momentum-trigger
   ```
4. **Webhook URL**: `https://your-n8n-domain/webhook/momentum-trigger`

## 🔐 Step 8: Security Configuration

### API Security
1. **Add Header Auth credential**
2. **Configuration**:
   ```
   Name: X-API-Key
   Value: your-secret-api-key
   ```
3. **Apply to HTTP Request node**

### Environment Variables
1. **Set in N8N**:
   ```
   MOMENTUM_API_URL=http://localhost:8000
   MOMENTUM_API_KEY=your-secret-key
   GOOGLE_SHEET_ID=your-sheet-id
   SLACK_CHANNEL=#trading
   ```

2. **Use in nodes**:
   ```
   URL: {{ $env.MOMENTUM_API_URL }}/run
   ```

## 📈 Step 9: Production Checklist

- [ ] All credentials created and tested
- [ ] Google Sheet headers match exactly
- [ ] Slack bot invited to channel
- [ ] API endpoint accessible from n8n
- [ ] Schedule set correctly (consider timezone)
- [ ] Error handling configured
- [ ] Notifications working
- [ ] Logs are being written
- [ ] Manual execution works
- [ ] All sensitive data in credentials (not hardcoded)

## 🔄 Step 10: Maintenance

### Regular Checks
1. **Weekly**: Verify data is logging correctly
2. **Monthly**: Check credential expiration
3. **Quarterly**: Review and optimize workflow

### Backup
1. **Export workflow**: Settings → Export
2. **Backup credentials**: Document settings
3. **Save configurations**: Keep copies of all settings

This detailed guide should resolve most import and configuration issues. Start with the minimal workflow and gradually add features once the basic version is working.
