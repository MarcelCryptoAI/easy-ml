# 🚀 DEPLOYMENT INSTRUCTIES

## ✅ KLAAR VOOR PRODUCTIE

Het systeem is volledig klaar voor gebruik. Geen dummy data, geen fallbacks - alleen echte trading met echte data.

## 🔑 VERPLICHTE ENVIRONMENT VARIABLES

Deze MOETEN ingesteld worden, anders start het systeem niet:

```
BYBIT_API_KEY=your_real_bybit_api_key
BYBIT_API_SECRET=your_real_bybit_secret
OPENAI_API_KEY=your_real_openai_key
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
SECRET_KEY=random_secret_key_here
BYBIT_TESTNET=true  # Zet op false voor live trading
```

## 📋 DEPLOYMENT STAPPEN

### 1. Railway Deployment
```bash
# Clone en ga naar directory
cd crypto-trading-platform

# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### 2. Environment Variables Instellen in Railway
- Ga naar Railway dashboard
- Open je project
- Ga naar Variables tab
- Voeg ALLE bovenstaande variabelen toe

### 3. Start Verificatie
Na deployment check `/health` endpoint:
```json
{
  "status": "healthy",
  "database_coins": 150,
  "bybit_connected": true,
  "available_symbols": 150,
  "account_balance": 1000.0,
  "trading_enabled": false
}
```

## ⚙️ SYSTEEM ACTIVATIE

### 1. Via Dashboard
- Open frontend URL
- Ga naar Dashboard
- Toggle "Auto Trading" aan
- Systeem start automatisch

### 2. Via API
```bash
curl -X POST "your-url/trading/toggle" -d '{"enable": true}'
```

## 🎯 WAT GEBEURT ER AUTOMATISCH

1. **Start**: Systeem valideert API keys
2. **Coins**: Laadt ALLE Bybit derivatives (geen limiet)
3. **Training**: ML worker start continue training cyclus
4. **Trading**: Bij 70%+ confidence van 3/4 modellen → trade
5. **AI**: OpenAI optimaliseert strategies gebaseerd op performance

## 📊 REAL-TIME MONITORING

- **Dashboard**: Live P&L, trades, posities
- **WebSocket**: Real-time updates
- **Health Check**: `/health` voor system status
- **Logs**: Railway logs voor debugging

## ⚠️ VEILIGHEID

- Start ALTIJD met `BYBIT_TESTNET=true`
- Test thoroughly voordat je live gaat
- Monitor de eerste trades closely
- Gebruik stop losses (standaard 1%)

## 🔧 INSTELLINGEN MANAGEMENT

Alle instellingen via webinterface aanpasbaar:
- Take profit/stop loss per coin
- Leverage settings
- Confidence thresholds
- Position sizing
- AI optimization on/off

Het systeem is 100% klaar voor gebruik!