# N8N Integration Guide for Momentum Strategy API

This guide shows how to integrate the momentum strategy API with n8n for automated rebalancing workflows.

## 🔄 Overview of N8N Integration

The momentum strategy API provides JSON output perfect for n8n automation. Here's how to implement automated rebalancing:

1. **Scheduled Strategy Runs** - Daily/weekly momentum analysis
2. **Data Processing** - Transform API output for broker APIs
3. **Portfolio Comparison** - Compare current vs recommended portfolio
4. **Order Generation** - Create buy/sell orders for rebalancing
5. **Execution & Monitoring** - Execute trades and monitor results

## 📋 N8N Workflow Examples

### 1. Basic Momentum Strategy Workflow

```json
{
  "name": "Momentum Strategy - Daily Rebalancing",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "cronExpression",
              "expression": "0 9 * * 1-5"
            }
          ]
        }
      },
      "name": "Schedule - Daily 9 AM Weekdays",
      "type": "n8n-nodes-base.cron",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://your-api-server:8000/run",
        "options": {
          "queryParameters": {
            "parameters": [
              {
                "name": "capital",
                "value": "500000"
              },
              {
                "name": "top_n",
                "value": "15"
              },
              {
                "name": "price_cap",
                "value": "3000"
              }
            ]
          }
        }
      },
      "name": "Get Momentum Analysis",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 3,
      "position": [450, 300]
    },
    {
      "parameters": {
        "jsCode": "// Process momentum strategy output\nconst data = $input.all()[0].json;\n\n// Extract top stocks\nconst topStocks = data.top || [];\nconst eligible = data.eligible || [];\nconst summary = data.summary || {};\n\n// Transform for portfolio management\nconst portfolio = topStocks.map(stock => ({\n  ticker: stock.Ticker,\n  targetWeight: stock['Allocation(%)'] || 0,\n  targetAmount: stock.Amount || 0,\n  targetQty: stock.Qty || 0,\n  currentPrice: stock.Price || 0,\n  momentumScore: stock.MomentumScore || 0,\n  signal: 'BUY',\n  priority: stock.Rank || 999\n}));\n\n// Create rebalancing summary\nconst rebalanceData = {\n  date: new Date().toISOString().split('T')[0],\n  totalCapital: summary.total_capital || 0,\n  investedAmount: summary.invested_amount || 0,\n  utilization: summary.utilization_pct || 0,\n  selectedStocks: summary.selected_stocks || 0,\n  eligibleStocks: summary.eligible_stocks || 0,\n  avgMomentumScore: summary.avg_momentum_score || 0,\n  portfolio: portfolio,\n  rawData: data\n};\n\nreturn [{ json: rebalanceData }];"
      },
      "name": "Process Strategy Output",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [650, 300]
    }
  ],
  "connections": {
    "Schedule - Daily 9 AM Weekdays": {
      "main": [
        [
          {
            "node": "Get Momentum Analysis",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Get Momentum Analysis": {
      "main": [
        [
          {
            "node": "Process Strategy Output",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 2. Portfolio Comparison & Rebalancing Workflow

```json
{
  "name": "Portfolio Rebalancing Logic",
  "nodes": [
    {
      "parameters": {
        "jsCode": "// Compare current portfolio with target\nconst targetPortfolio = $input.all()[0].json.portfolio;\nconst currentHoldings = $input.all()[1].json.holdings; // From broker API\n\nconst rebalanceOrders = [];\nconst currentMap = new Map();\n\n// Map current holdings\ncurrentHoldings.forEach(holding => {\n  currentMap.set(holding.ticker, {\n    currentQty: holding.quantity || 0,\n    currentValue: holding.marketValue || 0\n  });\n});\n\n// Generate rebalancing orders\ntargetPortfolio.forEach(target => {\n  const current = currentMap.get(target.ticker) || { currentQty: 0, currentValue: 0 };\n  const qtyDiff = target.targetQty - current.currentQty;\n  \n  if (Math.abs(qtyDiff) > 0) {\n    rebalanceOrders.push({\n      ticker: target.ticker,\n      action: qtyDiff > 0 ? 'BUY' : 'SELL',\n      quantity: Math.abs(qtyDiff),\n      targetQty: target.targetQty,\n      currentQty: current.currentQty,\n      currentPrice: target.currentPrice,\n      estimatedValue: Math.abs(qtyDiff) * target.currentPrice,\n      priority: target.priority,\n      momentumScore: target.momentumScore\n    });\n  }\n});\n\n// Handle positions to exit (not in target portfolio)\ncurrentMap.forEach((current, ticker) => {\n  const inTarget = targetPortfolio.find(t => t.ticker === ticker);\n  if (!inTarget && current.currentQty > 0) {\n    rebalanceOrders.push({\n      ticker: ticker,\n      action: 'SELL',\n      quantity: current.currentQty,\n      targetQty: 0,\n      currentQty: current.currentQty,\n      reason: 'EXIT_POSITION',\n      priority: 1 // High priority for exits\n    });\n  }\n});\n\n// Sort by priority (exits first, then buys)\nrebalanceOrders.sort((a, b) => {\n  if (a.action === 'SELL' && b.action === 'BUY') return -1;\n  if (a.action === 'BUY' && b.action === 'SELL') return 1;\n  return a.priority - b.priority;\n});\n\nreturn [{\n  json: {\n    rebalanceDate: new Date().toISOString(),\n    totalOrders: rebalanceOrders.length,\n    buyOrders: rebalanceOrders.filter(o => o.action === 'BUY').length,\n    sellOrders: rebalanceOrders.filter(o => o.action === 'SELL').length,\n    orders: rebalanceOrders\n  }\n}];"
      },
      "name": "Generate Rebalance Orders",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [450, 400]
    },
    {
      "parameters": {
        "conditions": {\n          \"options\": {\n            \"caseSensitive\": true,\n            \"leftValue\": \"\",\n            \"typeValidation\": \"strict\"\n          },\n          \"conditions\": [\n            {\n              \"leftValue\": \"={{ $json.totalOrders }}\",\n              \"rightValue\": 0,\n              \"operator\": {\n                \"type\": \"number\",\n                \"operation\": \"gt\"\n              }\n            }\n          ],\n          \"combinator\": \"and\"\n        }\n      },\n      "name": "Check if Rebalancing Needed",\n      "type": "n8n-nodes-base.if",\n      "typeVersion": 2,\n      "position": [650, 400]\n    }\n  ]\n}\n```\n\n## 🔗 Webhook Integration\n\nCreate a webhook endpoint in your API for n8n integration:\n\n```python\n# Add to main.py\nfrom fastapi import BackgroundTasks\nfrom typing import Optional\n\n@app.post(\"/webhook/rebalance\")\nasync def webhook_rebalance(\n    background_tasks: BackgroundTasks,\n    webhook_data: dict,\n    capital: Optional[float] = None,\n    top_n: Optional[int] = None\n):\n    \"\"\"\n    Webhook endpoint for n8n rebalancing automation\n    \"\"\"\n    try:\n        # Extract parameters from webhook\n        _capital = capital or webhook_data.get('capital', cfg['capital'])\n        _top_n = top_n or webhook_data.get('top_n', cfg['top_n'])\n        \n        # Run strategy analysis\n        result = run(capital=_capital, top_n=_top_n)\n        \n        # Add webhook metadata\n        result['webhook'] = {\n            'triggered_at': datetime.now().isoformat(),\n            'source': webhook_data.get('source', 'n8n'),\n            'execution_id': webhook_data.get('execution_id')\n        }\n        \n        return result\n        \n    except Exception as e:\n        logger.error(f\"Webhook error: {e}\")\n        raise HTTPException(status_code=500, detail=str(e))\n\n@app.get(\"/webhook/status/{execution_id}\")\nasync def webhook_status(execution_id: str):\n    \"\"\"\n    Check status of webhook execution\n    \"\"\"\n    # Implementation for tracking execution status\n    return {\"execution_id\": execution_id, \"status\": \"completed\"}\n```\n\n## 📊 Google Sheets Integration\n\nExample n8n workflow for Google Sheets logging:\n\n```json\n{\n  \"name\": \"Log to Google Sheets\",\n  \"nodes\": [\n    {\n      \"parameters\": {\n        \"operation\": \"appendOrUpdate\",\n        \"documentId\": \"your-google-sheet-id\",\n        \"sheetName\": \"Portfolio_Tracking\",\n        \"columns\": {\n          \"mappingMode\": \"defineBelow\",\n          \"value\": {\n            \"Date\": \"={{ $json.rebalanceDate }}\",\n            \"Total_Orders\": \"={{ $json.totalOrders }}\",\n            \"Buy_Orders\": \"={{ $json.buyOrders }}\",\n            \"Sell_Orders\": \"={{ $json.sellOrders }}\",\n            \"Total_Capital\": \"={{ $json.totalCapital }}\",\n            \"Utilization\": \"={{ $json.utilization }}\",\n            \"Avg_Momentum_Score\": \"={{ $json.avgMomentumScore }}\"\n          }\n        },\n        \"options\": {}\n      },\n      \"name\": \"Log Portfolio Summary\",\n      \"type\": \"n8n-nodes-base.googleSheets\",\n      \"typeVersion\": 4,\n      \"position\": [650, 500]\n    },\n    {\n      \"parameters\": {\n        \"operation\": \"clear\",\n        \"documentId\": \"your-google-sheet-id\",\n        \"sheetName\": \"Current_Holdings\",\n        \"options\": {}\n      },\n      \"name\": \"Clear Previous Holdings\",\n      \"type\": \"n8n-nodes-base.googleSheets\",\n      \"typeVersion\": 4,\n      \"position\": [450, 600]\n    },\n    {\n      \"parameters\": {\n        \"operation\": \"appendOrUpdate\",\n        \"documentId\": \"your-google-sheet-id\",\n        \"sheetName\": \"Current_Holdings\",\n        \"columns\": {\n          \"mappingMode\": \"defineBelow\",\n          \"value\": {\n            \"Rank\": \"={{ $json.Rank }}\",\n            \"Ticker\": \"={{ $json.Ticker }}\",\n            \"Price\": \"={{ $json.Price }}\",\n            \"Target_Qty\": \"={{ $json.Qty }}\",\n            \"Target_Amount\": \"={{ $json.Amount }}\",\n            \"Allocation_Pct\": \"={{ $json['Allocation(%)'] }}\",\n            \"Momentum_Score\": \"={{ $json.MomentumScore }}\",\n            \"Return_6M\": \"={{ $json['6M_Return(%)'] }}\",\n            \"Return_12M_ex1\": \"={{ $json['12M_ex1_Return(%)'] }}\",\n            \"Volatility\": \"={{ $json['Volatility(%)'] }}\"\n          }\n        },\n        \"options\": {}\n      },\n      \"name\": \"Update Current Holdings\",\n      \"type\": \"n8n-nodes-base.googleSheets\",\n      \"typeVersion\": 4,\n      \"position\": [650, 600]\n    }\n  ]\n}\n```\n\n## 📱 Broker API Integration Examples\n\n### Zerodha Kite Integration\n\n```json\n{\n  \"name\": \"Zerodha Order Execution\",\n  \"nodes\": [\n    {\n      \"parameters\": {\n        \"jsCode\": \"// Format orders for Zerodha Kite API\\nconst orders = $input.all()[0].json.orders;\\n\\nconst kiteOrders = orders.map(order => ({\\n  tradingsymbol: order.ticker.replace('.NS', ''),\\n  exchange: 'NSE',\\n  transaction_type: order.action,\\n  quantity: order.quantity,\\n  order_type: 'MARKET',\\n  product: 'CNC',\\n  validity: 'DAY',\\n  tag: `momentum_${new Date().toISOString().split('T')[0]}`\\n}));\\n\\nreturn kiteOrders.map(order => ({ json: order }));\"\n      },\n      \"name\": \"Format Kite Orders\",\n      \"type\": \"n8n-nodes-base.code\",\n      \"typeVersion\": 2,\n      \"position\": [450, 300]\n    },\n    {\n      \"parameters\": {\n        \"url\": \"https://api.kite.trade/orders/regular\",\n        \"authentication\": \"genericCredentialType\",\n        \"genericAuthType\": \"httpHeaderAuth\",\n        \"httpMethod\": \"POST\",\n        \"sendBody\": true,\n        \"specifyBodyContentType\": true,\n        \"bodyContentType\": \"form-urlencoded\",\n        \"bodyParameters\": {\n          \"parameters\": [\n            {\n              \"name\": \"tradingsymbol\",\n              \"value\": \"={{ $json.tradingsymbol }}\"\n            },\n            {\n              \"name\": \"exchange\",\n              \"value\": \"={{ $json.exchange }}\"\n            },\n            {\n              \"name\": \"transaction_type\",\n              \"value\": \"={{ $json.transaction_type }}\"\n            },\n            {\n              \"name\": \"quantity\",\n              \"value\": \"={{ $json.quantity }}\"\n            },\n            {\n              \"name\": \"order_type\",\n              \"value\": \"={{ $json.order_type }}\"\n            },\n            {\n              \"name\": \"product\",\n              \"value\": \"={{ $json.product }}\"\n            }\n          ]\n        }\n      },\n      \"name\": \"Place Kite Order\",\n      \"type\": \"n8n-nodes-base.httpRequest\",\n      \"typeVersion\": 3,\n      \"position\": [650, 300]\n    }\n  ]\n}\n```\n\n### ICICI Direct Integration\n\n```json\n{\n  \"name\": \"ICICI Direct Order\",\n  \"parameters\": {\n    \"url\": \"https://api.icicidirect.com/equity/orders\",\n    \"authentication\": \"genericCredentialType\",\n    \"httpMethod\": \"POST\",\n    \"sendBody\": true,\n    \"bodyContentType\": \"json\",\n    \"jsonBody\": \"={\\n  \\\"instrument_token\\\": \\\"{{ $json.ticker }}\\\",\\n  \\\"quantity\\\": \\\"{{ $json.quantity }}\\\",\\n  \\\"price\\\": 0,\\n  \\\"order_type\\\": \\\"MARKET\\\",\\n  \\\"validity\\\": \\\"DAY\\\",\\n  \\\"product\\\": \\\"DELIVERY\\\",\\n  \\\"transaction_type\\\": \\\"{{ $json.action }}\\\"\\n}\"\n  }\n}\n```\n\n## 🔔 Alerting & Monitoring\n\n### Slack Notifications\n\n```json\n{\n  \"name\": \"Slack Alert\",\n  \"parameters\": {\n    \"channel\": \"#trading-alerts\",\n    \"text\": \"🔄 *Momentum Strategy Rebalancing*\\n\\n📊 *Summary:*\\n• Date: {{ $json.rebalanceDate }}\\n• Total Orders: {{ $json.totalOrders }}\\n• Buy Orders: {{ $json.buyOrders }}\\n• Sell Orders: {{ $json.sellOrders }}\\n• Capital Utilization: {{ $json.utilization }}%\\n\\n💹 *Top Holdings:*\\n{{ $json.orders.slice(0,5).map(o => `• ${o.ticker}: ${o.action} ${o.quantity} shares`).join('\\\\n') }}\\n\\n🔗 <https://your-dashboard.com|View Dashboard>\",\n    \"otherOptions\": {\n      \"mrkdwn\": true\n    }\n  },\n  \"type\": \"n8n-nodes-base.slack\"\n}\n```\n\n### Email Notifications\n\n```json\n{\n  \"name\": \"Email Report\",\n  \"parameters\": {\n    \"fromEmail\": \"trading@yourcompany.com\",\n    \"toEmail\": \"portfolio@yourcompany.com\",\n    \"subject\": \"Daily Momentum Strategy Report - {{ $now.format('YYYY-MM-DD') }}\",\n    \"html\": \"<h2>Momentum Strategy Rebalancing Report</h2>\\n<table border='1'>\\n<tr><th>Ticker</th><th>Action</th><th>Quantity</th><th>Price</th><th>Value</th></tr>\\n{{ $json.orders.map(o => `<tr><td>${o.ticker}</td><td>${o.action}</td><td>${o.quantity}</td><td>${o.currentPrice}</td><td>${o.estimatedValue}</td></tr>`).join('') }}\\n</table>\"\n  },\n  \"type\": \"n8n-nodes-base.emailSend\"\n}\n```\n\n## ⏰ Scheduling Options\n\n### 1. Daily Rebalancing (Conservative)\n```javascript\n// Cron: 0 9 * * 1-5 (9 AM weekdays)\n{\n  \"rule\": {\n    \"interval\": [{\n      \"field\": \"cronExpression\",\n      \"expression\": \"0 9 * * 1-5\"\n    }]\n  }\n}\n```\n\n### 2. Weekly Rebalancing (Balanced)\n```javascript\n// Cron: 0 9 * * 1 (9 AM Mondays)\n{\n  \"rule\": {\n    \"interval\": [{\n      \"field\": \"cronExpression\",\n      \"expression\": \"0 9 * * 1\"\n    }]\n  }\n}\n```\n\n### 3. Monthly Rebalancing (Aggressive)\n```javascript\n// Cron: 0 9 1 * * (9 AM 1st of month)\n{\n  \"rule\": {\n    \"interval\": [{\n      \"field\": \"cronExpression\",\n      \"expression\": \"0 9 1 * *\"\n    }]\n  }\n}\n```\n\n## 🛡️ Risk Management in N8N\n\n### Position Size Validation\n\n```javascript\n// Risk check before order execution\nconst orders = $input.all()[0].json.orders;\nconst totalCapital = $input.all()[0].json.totalCapital;\nconst maxPositionSize = 0.1; // 10% max per position\n\nconst validatedOrders = orders.filter(order => {\n  const positionSize = order.estimatedValue / totalCapital;\n  if (positionSize > maxPositionSize) {\n    console.log(`Position size too large for ${order.ticker}: ${positionSize * 100}%`);\n    return false;\n  }\n  return true;\n});\n\nreturn [{ json: { validatedOrders, rejectedCount: orders.length - validatedOrders.length } }];\n```\n\n### Market Hours Check\n\n```javascript\n// Check if market is open before placing orders\nconst now = new Date();\nconst istTime = new Date(now.toLocaleString(\"en-US\", {timeZone: \"Asia/Kolkata\"}));\nconst hour = istTime.getHours();\nconst day = istTime.getDay();\n\nconst isMarketHours = (\n  day >= 1 && day <= 5 && // Monday to Friday\n  hour >= 9 && hour <= 15  // 9 AM to 3:30 PM IST\n);\n\nif (!isMarketHours) {\n  throw new Error('Market is closed. Orders will be queued for next session.');\n}\n\nreturn $input.all();\n```\n\n## 📈 Performance Tracking\n\n### Portfolio Performance Calculation\n\n```javascript\n// Calculate portfolio performance\nconst currentPortfolio = $input.all()[0].json;\nconst previousPortfolio = $input.all()[1].json; // From database/sheets\n\nconst performance = {\n  date: new Date().toISOString().split('T')[0],\n  totalValue: currentPortfolio.reduce((sum, holding) => \n    sum + (holding.quantity * holding.currentPrice), 0),\n  totalReturn: 0, // Calculate from previous value\n  dayChange: 0,\n  weekChange: 0,\n  monthChange: 0,\n  positions: currentPortfolio.length,\n  topPerformer: null,\n  worstPerformer: null\n};\n\n// Calculate individual stock performance\nconst stockPerformance = currentPortfolio.map(stock => ({\n  ticker: stock.ticker,\n  dayReturn: stock.dayChange || 0,\n  weekReturn: stock.weekChange || 0,\n  monthReturn: stock.monthChange || 0,\n  momentumScore: stock.momentumScore\n}));\n\nperformance.topPerformer = stockPerformance.reduce((best, current) => \n  current.dayReturn > best.dayReturn ? current : best);\n\nperformance.worstPerformer = stockPerformance.reduce((worst, current) => \n  current.dayReturn < worst.dayReturn ? current : worst);\n\nreturn [{ json: performance }];\n```\n\n## 🔄 Complete Rebalancing Workflow\n\nHere's a complete n8n workflow JSON that you can import:\n