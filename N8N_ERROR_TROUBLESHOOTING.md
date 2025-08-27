# N8N Error Troubleshooting Guide

This guide addresses specific error messages you might encounter when importing or configuring the momentum strategy workflow.

## 🚨 Import Errors

### Error: "Workflow could not be imported"
**Symptoms:**
- Import dialog shows error message
- JSON file appears corrupted

**Solutions:**
1. **Check JSON Validity:**
   ```bash
   # Validate JSON online or with tool
   cat n8n_workflow.json | python -m json.tool
   ```

2. **Try Minimal Workflow First:**
   - Use `n8n_minimal_working.json` instead
   - Import this simpler version first

3. **Manual Import Alternative:**
   ```
   1. Copy workflow JSON content
   2. Go to n8n → New Workflow
   3. Click on workflow settings (gear icon)
   4. Paste JSON in import field
   ```

### Error: "Unknown node type"
**Symptoms:**
- Some nodes show as "Unknown" after import
- Workflow has missing node types

**Solutions:**
1. **Check N8N Version:**
   ```
   Minimum required: n8n v1.0+
   Check: Help → About in n8n interface
   ```

2. **Update N8N:**
   ```bash
   # If using npm
   npm update n8n -g
   
   # If using Docker
   docker pull n8nio/n8n:latest
   ```

3. **Use Compatible Nodes:**
   - Replace unknown nodes with compatible alternatives
   - Refer to manual node creation guide

### Error: "Credential type not found"
**Symptoms:**
- Nodes show red exclamation marks
- Error about missing credentials

**Solutions:**
1. **Create Credentials First:**
   ```
   BEFORE importing workflow:
   1. Go to Settings → Credentials
   2. Create all required credentials
   3. Test each credential
   4. Then import workflow
   ```

2. **Required Credentials List:**
   - Google Sheets OAuth2 API
   - Slack OAuth2 API  
   - SMTP (for email)
   - HTTP Header Auth (for broker APIs)

## ⚙️ Configuration Errors

### Error: "Google Sheets permission denied"
**Full Error Message:**
```
Error: Insufficient Permission: Request had insufficient authentication scopes.
```

**Solutions:**
1. **Re-create Google Cloud Project:**
   ```
   1. Go to console.cloud.google.com
   2. Create new project: "n8n-momentum-strategy"
   3. Enable Google Sheets API
   4. Create OAuth 2.0 credentials
   ```

2. **Correct OAuth Scopes:**
   ```
   Add these scopes in Google Cloud Console:
   - https://www.googleapis.com/auth/spreadsheets
   - https://www.googleapis.com/auth/drive.file
   ```

3. **OAuth Setup Steps:**
   ```
   1. OAuth consent screen → External
   2. Add your email as test user
   3. Scopes → Add Google Sheets scopes
   4. Create OAuth client ID → Web application
   5. Add redirect URI: https://your-n8n-domain/rest/oauth2-credential/callback
   ```

4. **N8N Credential Configuration:**
   ```
   Type: Google Sheets OAuth2 API
   Client ID: [from Google Cloud Console]
   Client Secret: [from Google Cloud Console]
   Scope: https://www.googleapis.com/auth/spreadsheets
   ```

### Error: "Slack channel not found"
**Full Error Message:**
```
Error: channel_not_found
```

**Solutions:**
1. **Check Channel Name:**
   ```
   Correct format: #trading
   Include the # symbol
   Case sensitive
   ```

2. **Invite Bot to Channel:**
   ```
   In Slack:
   1. Go to #trading channel
   2. Type: /invite @your-bot-name
   3. Or manually add via channel settings
   ```

3. **Bot Permissions:**
   ```
   Required OAuth scopes:
   - chat:write
   - channels:read
   - groups:read (for private channels)
   ```

4. **Create Slack App Properly:**
   ```
   1. Go to api.slack.com
   2. Create App → From scratch
   3. Choose workspace
   4. OAuth & Permissions → Add scopes
   5. Install to workspace
   6. Copy "Bot User OAuth Token"
   ```

### Error: "API connection refused"
**Full Error Message:**
```
Error: connect ECONNREFUSED 127.0.0.1:8000
```

**Solutions:**
1. **Verify API is Running:**
   ```bash
   # Test locally
   curl http://localhost:8000/health
   
   # Should return:
   {"status": "healthy", "timestamp": "..."}
   ```

2. **Network Configuration:**
   ```
   If n8n is remote, replace localhost:
   - localhost:8000 → your-server-ip:8000
   - 127.0.0.1:8000 → your-server-ip:8000
   ```

3. **Firewall Settings:**
   ```bash
   # Check if port 8000 is accessible
   telnet your-server-ip 8000
   
   # If needed, open port
   sudo ufw allow 8000
   ```

4. **Docker Network Issues:**
   ```bash
   # If API in Docker, use container name
   http://momentum-api:8000/run
   
   # Or use host network
   docker run --network host momentum-api
   ```

### Error: "Invalid expression"
**Full Error Message:**
```
Error: Invalid expression [SyntaxError: Unexpected token ...]
```

**Solutions:**
1. **Expression Syntax:**
   ```javascript
   // Correct format
   {{ $json.fieldName }}
   
   // Common mistakes to avoid
   { $json.fieldName }     // Missing outer braces
   {{ json.fieldName }}    // Missing $
   {{ $json.field-name }}  // Hyphens in field names
   ```

2. **Safe Field Access:**
   ```javascript
   // Use optional chaining for safety
   {{ $json.summary?.total_capital || 0 }}
   {{ $json.top?.[0]?.Ticker || '' }}
   ```

3. **Code Node Errors:**
   ```javascript
   // Always check input exists
   const input = $input.all()[0]?.json;
   if (!input) {
     throw new Error('No input data');
   }
   ```

## 🔧 Execution Errors

### Error: "Workflow execution failed"
**Symptoms:**
- Red error indicators on nodes
- Execution stops mid-workflow

**Debugging Steps:**
1. **Execute Nodes Individually:**
   ```
   1. Click on first node
   2. Click "Execute Node"
   3. Check output data
   4. Repeat for each node
   ```

2. **Check Node Outputs:**
   ```
   Each node should show:
   - Green checkmark (success)
   - Output data in JSON format
   - No error messages
   ```

3. **Common Output Issues:**
   ```javascript
   // HTTP Request should return
   {
     "summary": { ... },
     "top": [ ... ],
     "eligible": [ ... ]
   }
   
   // Code node should return
   {
     "executionDate": "2024-01-15",
     "totalCapital": 100000,
     "status": "SUCCESS"
   }
   ```

### Error: "Timeout exceeded"
**Full Error Message:**
```
Error: Workflow execution timed out
```

**Solutions:**
1. **Increase Timeout:**
   ```
   HTTP Request node → Options → Timeout: 120000 (2 minutes)
   ```

2. **Optimize API Performance:**
   ```
   Reduce parameters:
   - capital: 50000 (smaller amount)
   - top_n: 3 (fewer stocks)
   - Test with recent date
   ```

3. **Check API Performance:**
   ```bash
   # Time the API call
   time curl "http://localhost:8000/run?capital=10000&top_n=3"
   ```

### Error: "Data transformation failed"
**Symptoms:**
- Code node shows errors
- Data format doesn't match expected structure

**Solutions:**
1. **Debug Data Structure:**
   ```javascript
   // Add debug logging in Code node
   console.log('Input structure:', JSON.stringify($input.all()[0].json, null, 2));
   
   // Check what API actually returns
   const input = $input.all()[0].json;
   console.log('Summary exists:', !!input.summary);
   console.log('Top exists:', !!input.top);
   ```

2. **Handle Missing Fields:**
   ```javascript
   // Defensive programming
   const summary = input?.summary || {};
   const topStocks = input?.top || [];
   const eligibleStocks = input?.eligible || [];
   
   // Provide defaults
   const result = {
     totalCapital: summary.total_capital || 0,
     selectedStocks: summary.selected_stocks || 0,
     utilization: summary.utilization_pct || 0,
     avgMomentumScore: summary.avg_momentum_score || 0
   };
   ```

## 🔐 Authentication Errors

### Error: "OAuth token expired"
**Solutions:**
1. **Refresh Google Credential:**
   ```
   1. Go to Settings → Credentials
   2. Find Google Sheets credential
   3. Click "Reconnect"
   4. Complete OAuth flow again
   ```

2. **Check Token Expiry:**
   ```
   Google tokens expire after 7 days of inactivity
   Regular use keeps them active
   ```

### Error: "Slack token invalid"
**Solutions:**
1. **Regenerate Slack Token:**
   ```
   1. Go to api.slack.com → Your App
   2. OAuth & Permissions
   3. Reinstall to workspace
   4. Copy new token to n8n
   ```

2. **Verify Bot Permissions:**
   ```
   Required scopes:
   - chat:write
   - channels:read
   ```

## 🧪 Testing & Validation

### Complete Testing Checklist

#### 1. Credential Testing
```
□ Google Sheets: Can create/edit sheets
□ Slack: Can post messages
□ API: Returns valid data
```

#### 2. Node Testing
```
□ Cron: Shows correct schedule
□ HTTP: Returns API data
□ Code: Processes data correctly
□ Sheets: Writes data successfully
□ Slack: Sends notification
```

#### 3. Data Flow Testing
```
□ API returns expected JSON structure
□ Code node transforms data correctly
□ Google Sheets receives correct format
□ Slack gets properly formatted message
```

#### 4. Error Handling Testing
```
□ API unavailable: Workflow handles gracefully
□ Invalid data: Code node doesn't crash
□ Google Sheets offline: Shows clear error
□ Slack unavailable: Continues execution
```

## 🔄 Recovery Procedures

### If Workflow Completely Broken
1. **Start Fresh:**
   ```
   1. Create new blank workflow
   2. Add nodes one by one
   3. Test each addition
   4. Use manual node creation
   ```

2. **Import Clean Version:**
   ```
   1. Use n8n_minimal_working.json
   2. Test basic functionality
   3. Add features incrementally
   ```

### If Credentials Lost
1. **Re-create All Credentials:**
   ```
   1. Delete old credentials
   2. Create new ones with same names
   3. Re-authenticate all services
   4. Update workflow nodes
   ```

### If Data Format Changed
1. **Update Code Node:**
   ```javascript
   // Log actual API response
   console.log('API Response:', JSON.stringify($input.all()[0].json, null, 2));
   
   // Adjust field mappings
   const summary = input.summary || input.data || input.result;
   ```

## 📞 Getting Help

### Before Asking for Help
1. **Export workflow JSON**
2. **Copy exact error messages**  
3. **Document steps that led to error**
4. **Test with minimal workflow first**

### Information to Include
```
1. N8N version
2. Installation method (cloud/docker/npm)
3. Exact error message
4. Node configuration screenshots
5. API response samples
6. Browser console errors (if any)
```

This troubleshooting guide should help resolve most issues you encounter during setup and configuration.