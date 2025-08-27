# Complete N8N Setup Guide for Momentum Strategy

This guide provides step-by-step instructions to implement automated rebalancing using n8n with the momentum strategy API.

## 🚀 Quick Setup Overview

1. **API Setup** - Deploy the momentum strategy API
2. **N8N Installation** - Set up n8n workflow automation
3. **Workflow Import** - Import the pre-built workflow
4. **Configuration** - Configure credentials and settings
5. **Testing** - Test the complete automation
6. **Production** - Deploy for live trading

## 📋 Prerequisites

### API Requirements
- Momentum Strategy API running (localhost:8000 or deployed)
- Python 3.8+ with all dependencies installed
- Valid stock universe data source

### N8N Requirements
- N8N installed (local, cloud, or self-hosted)
- Google Sheets access (for logging)
- Slack workspace (for notifications)
- Email SMTP settings
- Broker API credentials (Zerodha, ICICI, etc.)

## 🛠️ Step 1: API Deployment

### Local Development
```bash
# Clone and setup
git clone <your-repo>
cd momentum-strategy
pip install -r requirements.txt

# Start API server
python run_server.py
# API will be available at http://localhost:8000
```

### Production Deployment
```bash
# Using Docker (recommended)
docker build -t momentum-api .
docker run -p 8000:8000 momentum-api

# Or using cloud deployment
# Deploy to Railway, Heroku, AWS, etc.
```

### Verify API
```bash
curl http://localhost:8000/health
curl "http://localhost:8000/run?capital=100000&top_n=5"
```

## 📊 Step 2: N8N Installation

### Option A: N8N Cloud (Recommended for beginners)
1. Visit [n8n.cloud](https://n8n.cloud)
2. Create account and workspace
3. Access your n8n instance

### Option B: Local Installation
```bash
# Using npm
npm install n8n -g
n8n start

# Using Docker
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n

# Access at http://localhost:5678
```

### Option C: Self-Hosted
```bash
# Using Docker Compose
version: '3.8'
services:
  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=your-password
    volumes:
      - n8n_data:/home/node/.n8n
volumes:
  n8n_data:
```

## 🔧 Step 3: Workflow Import

### Import the Complete Workflow
1. **Download Workflow**: Use the `n8n_complete_workflow.json` file
2. **Import in N8N**:
   - Go to n8n dashboard
   - Click "Import from File"
   - Select the workflow JSON file
   - Click "Import"

### Manual Setup Alternative
If importing doesn't work, create nodes manually:

1. **Cron Trigger** - Schedule (Monday 9 AM)
2. **HTTP Request** - Call momentum API
3. **Code Node** - Process strategy output
4. **Conditional Nodes** - Check results and market hours
5. **Google Sheets** - Log data
6. **Slack/Email** - Send notifications
7. **Broker API** - Execute orders

## ⚙️ Step 4: Configuration

### 4.1 API Connection Settings
```javascript
// In HTTP Request nodes, update URL
URL: "http://your-api-domain:8000/run"
// or for local testing
URL: "http://localhost:8000/run"
```

### 4.2 Google Sheets Setup
1. **Create Google Sheets Document**:
   - Create new Google Sheet
   - Add sheets: "Rebalancing_Log", "Order_Execution_Log", "Portfolio_Tracking"
   - Copy the Sheet ID from URL

2. **Google Sheets Credentials**:
   - Go to N8N Settings > Credentials
   - Add "Google Sheets OAuth2 API"
   - Follow authentication flow
   - Set permissions for Sheets access

3. **Configure Sheet Nodes**:
   ```javascript
   // Update documentId in all Google Sheets nodes
   documentId: "your-google-sheet-id-here"
   ```

### 4.3 Slack Integration
1. **Create Slack App**:
   - Go to [api.slack.com](https://api.slack.com)
   - Create new app
   - Add to workspace
   - Get Bot Token

2. **N8N Slack Credentials**:
   - Add "Slack OAuth2 API" credential
   - Enter Bot Token
   - Test connection

3. **Configure Channel**:
   ```javascript
   // Update channel in Slack nodes
   channel: "#trading-alerts"
   ```

### 4.4 Email Configuration
1. **SMTP Settings**:
   - Add "SMTP" credential in N8N
   - Configure your email provider settings
   - Test email sending

2. **Email Recipients**:
   ```javascript
   // Update email addresses
   toEmail: "your-email@domain.com"
   fromEmail: "trading@yourdomain.com"
   ```

### 4.5 Broker API Setup

#### Zerodha Kite Integration
1. **Get Kite API Credentials**:
   - Register for Kite API
   - Get API key and secret
   - Set up authentication

2. **Configure API Credentials**:
   ```javascript
   // Add HTTP Header Auth credential
   // Header: Authorization
   // Value: token api_key:access_token
   ```

#### Other Brokers
- **ICICI Direct**: Similar setup with their API endpoints
- **Angel Broking**: Use Angel API documentation
- **Upstox**: Configure Upstox API credentials

### 4.6 Workflow Settings
Set these variables in N8N workflow settings:

```javascript
{
  "capital": 500000,
  "top_n": 15,
  "price_cap": 3000,
  "googleSheetId": "your-sheet-id",
  "alertEmail": "alerts@yourdomain.com"
}
```

## 📅 Step 5: Scheduling Configuration

### Schedule Options
```javascript
// Daily (weekdays only)
"0 9 * * 1-5"  // 9 AM Monday to Friday

// Weekly (Mondays)
"0 9 * * 1"    // 9 AM every Monday

// Monthly (1st of month)
"0 9 1 * *"    // 9 AM 1st day of month

// Custom (twice a week)
"0 9 * * 1,4"  // 9 AM Monday and Thursday
```

### Market Hours Validation
The workflow automatically checks:
- Market open: 9:15 AM - 3:30 PM IST
- Trading days: Monday - Friday
- Holidays: Manual override required

## 🧪 Step 6: Testing

### 6.1 Test Individual Components

#### Test API Connection
```bash
# In n8n, create simple workflow:
# HTTP Request → http://localhost:8000/health
# Should return: {"status": "healthy", ...}
```

#### Test Strategy Call
```bash
# HTTP Request → http://localhost:8000/run?capital=10000&top_n=3
# Should return strategy results with top stocks
```

#### Test Webhook
```bash
# POST to http://localhost:8000/webhook/rebalance
# Body: {"source": "n8n", "capital": 10000}
```

### 6.2 Test Complete Workflow
1. **Manual Execution**:
   - Go to workflow in n8n
   - Click "Execute Workflow"
   - Monitor each node execution
   - Check outputs and logs

2. **Scheduled Test**:
   - Set schedule to run in 5 minutes
   - Monitor execution
   - Verify all integrations work

### 6.3 Validate Outputs
- **Google Sheets**: Check data is logged correctly
- **Slack**: Verify notification received
- **Email**: Check email report format
- **Orders**: Validate order format (don't execute yet)

## 🔒 Step 7: Security & Risk Management

### 7.1 API Security
```python
# Add API key authentication
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key != "your-secret-api-key":
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    return await call_next(request)
```

### 7.2 Position Size Limits
```javascript
// In validation node
const maxPositionSize = 0.15; // 15% maximum
const maxDailyTurnover = 0.25; // 25% daily turnover limit
```

### 7.3 Trading Safeguards
- **Dry Run Mode**: Test without actual trades
- **Order Limits**: Maximum orders per day
- **Price Validation**: Check prices before execution
- **Manual Override**: Emergency stop mechanism

## 🚀 Step 8: Production Deployment

### 8.1 Environment Setup
```bash
# Production API deployment
export ENVIRONMENT=production
export API_KEY=your-production-api-key
export DATABASE_URL=your-production-db

# Start with production config
python main.py
```

### 8.2 Monitoring Setup
1. **Health Checks**: Monitor API uptime
2. **Error Alerts**: Set up error notifications
3. **Performance Monitoring**: Track execution times
4. **Trade Logging**: Comprehensive audit trail

### 8.3 Backup & Recovery
- **Configuration Backup**: Export n8n workflows
- **Data Backup**: Google Sheets automatic backup
- **Rollback Plan**: Quick workflow disable method

## 📊 Step 9: Dashboard & Monitoring

### 9.1 Google Sheets Dashboard
Create charts and pivot tables:
```
=QUERY(Rebalancing_Log!A:M, "SELECT Date, Total_Orders, Utilization_Pct ORDER BY Date DESC LIMIT 30")
```

### 9.2 Slack Dashboard
Custom Slack commands:
```javascript
// Daily summary command
/momentum-summary
// Returns recent performance metrics
```

### 9.3 Email Reports
Automated weekly/monthly reports with:
- Portfolio performance
- Trade statistics
- Risk metrics
- Momentum score trends

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Failed
```bash
# Check API status
curl http://localhost:8000/health

# Check n8n network access
# Ensure localhost accessible from n8n
```

#### 2. Data Not Updating
- Verify API credentials
- Check schedule configuration
- Review error logs in n8n

#### 3. Orders Not Executing
- Confirm market hours
- Verify broker API credentials
- Check position size limits

#### 4. Missing Notifications
- Test Slack/email credentials
- Verify channel/email settings
- Check workflow connections

### Debug Steps
1. **Enable Verbose Logging**: Set log level to DEBUG
2. **Test Each Node**: Execute nodes individually
3. **Check API Logs**: Monitor API server logs
4. **Validate Data Flow**: Trace data through workflow

## 📈 Advanced Features

### 1. Multi-Strategy Support
```javascript
// Run multiple strategies
const strategies = ['momentum', 'mean_reversion', 'trend_following'];
// Execute and combine results
```

### 2. Portfolio Optimization
```javascript
// Risk parity weighting
// Maximum diversification
// Minimum variance optimization
```

### 3. Real-time Monitoring
```javascript
// WebSocket connections for live data
// Real-time portfolio updates
// Live P&L tracking
```

### 4. Machine Learning Integration
```javascript
// Sentiment analysis
// News impact assessment
// Predictive modeling
```

## 📝 Example Workflows

### Simple Daily Rebalancing
1. **9 AM trigger** → Get momentum data
2. **Process results** → Generate orders
3. **Market hours check** → Validate timing
4. **Execute trades** → Place orders
5. **Log & notify** → Record and alert

### Advanced Risk-Managed Rebalancing
1. **Pre-market analysis** → Multiple data sources
2. **Risk assessment** → VaR and stress testing
3. **Position sizing** → Kelly criterion optimization
4. **Order optimization** → TWAP/VWAP execution
5. **Post-trade analysis** → Performance attribution

### Weekly Portfolio Review
1. **Monday morning** → Full strategy run
2. **Mid-week check** → Risk monitoring
3. **Friday close** → Performance review
4. **Weekend analysis** → Strategy evaluation

This comprehensive setup guide provides everything needed to implement automated momentum strategy rebalancing with n8n. Start with the basic setup and gradually add advanced features as you become comfortable with the system.

## 🎯 Next Steps

1. **Start Small**: Begin with paper trading
2. **Monitor Closely**: Watch first few executions
3. **Iterate & Improve**: Refine based on results
4. **Scale Gradually**: Increase capital allocation
5. **Add Features**: Implement advanced functionality

Remember to always test thoroughly before deploying with real money!