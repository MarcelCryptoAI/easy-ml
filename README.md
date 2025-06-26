# Crypto Trading ML Platform

Een volledig autonome crypto trading platform die machine learning gebruikt voor het analyseren en handelen van cryptocurrency derivatives op Bybit.

## Functies

### 🤖 Machine Learning
- **4 ML Modellen**: LSTM, Random Forest, SVM, en Neural Network
- **Continue Training**: Alle munten worden continu getraind voor real-time voorspellingen
- **Feature Engineering**: 50+ technische indicatoren en marktfeatures
- **Confidence Scoring**: Elk model geeft een confidence score voor besluitvorming

### 📈 Automatische Trading
- **Bybit API Integratie**: Directe verbinding met Bybit derivatives
- **Autonomous Trading**: Volledig automatische order plaatsing
- **Risk Management**: Configureerbare take profit, stop loss en leverage
- **Position Management**: Maximaal aantal posities en position sizing

### 🧠 AI Optimalisatie
- **OpenAI Integratie**: GPT-4 analyseert trading performance
- **Strategy Optimization**: AI past trading parameters automatisch aan
- **Performance Analysis**: Continue optimalisatie gebaseerd op resultaten
- **Conservative Approach**: Veilige en conservatieve aanpassingen

### 📊 Real-time Dashboard
- **Live Trading Data**: Real-time overzicht van alle trades en posities
- **ML Predictions**: Visualisatie van alle model voorspellingen per munt
- **Performance Metrics**: P&L tracking, win rate, en performance grafieken
- **WebSocket Updates**: Live updates van trades en market data

## API Endpoints

### Trading
- `GET /coins` - Lijst van beschikbare munten
- `GET /trades` - Trading geschiedenis
- `GET /positions` - Actieve posities
- `POST /trading/toggle` - Trading aan/uit zetten

### Machine Learning
- `GET /predictions/{symbol}` - ML voorspellingen voor munt
- `POST /optimize/{symbol}` - Optimaliseer strategie voor munt
- `POST /optimize/batch` - Optimaliseer alle strategieën

### Strategy Management
- `GET /strategy/{symbol}` - Huidige strategie voor munt
- `PUT /strategy/{symbol}` - Update strategie parameters

## Installation

1. Clone repository
2. Copy `.env.example` naar `.env` en vul API keys in
3. Deploy naar Railway of run lokaal met Docker
4. Toegang tot frontend via deployed URL

## Trading Strategie

- **Take Profit**: 2% (aanpasbaar per munt)
- **Stop Loss**: 1% (aanpasbaar per munt)
- **Confidence Threshold**: 70% (3 van 4 modellen moet overeenstemmen)
- **AI Optimalisatie**: OpenAI analyseert performance en past parameters aan