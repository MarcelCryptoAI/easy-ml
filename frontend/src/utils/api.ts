import axios from 'axios';
import { Coin, Trade, Position, TradingStrategy, MLPrediction } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for better error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

export const tradingApi = {
  // Coins
  getCoins: (): Promise<Coin[]> => 
    api.get('/coins').then(res => res.data),

  // Predictions
  getPredictions: (symbol: string): Promise<MLPrediction[]> =>
    api.get(`/predictions/${symbol}`).then(res => res.data),

  // Trades
  getTrades: (status?: string): Promise<Trade[]> =>
    api.get('/trades', { params: { status } }).then(res => res.data),

  // Positions
  getPositions: (): Promise<{ success: boolean; positions: Position[] }> =>
    api.get('/positions').then(res => res.data),

  // Strategy
  getStrategy: (symbol: string): Promise<TradingStrategy> =>
    api.get(`/strategy/${symbol}`).then(res => res.data),

  updateStrategy: (symbol: string, strategy: Partial<TradingStrategy>): Promise<{ success: boolean; message: string }> =>
    api.put(`/strategy/${symbol}`, strategy).then(res => res.data),

  // Trading controls
  toggleTrading: (enable: boolean): Promise<{ success: boolean; message: string }> =>
    api.post('/trading/toggle', { enable }).then(res => res.data),

  // AI Optimization
  optimizeStrategy: (symbol: string): Promise<any> =>
    api.post(`/optimize/${symbol}`).then(res => res.data),

  batchOptimize: (): Promise<any> =>
    api.post('/optimize/batch').then(res => res.data),

  // AI Recommendations
  getRecommendation: (symbol: string): Promise<any> =>
    api.get(`/recommendations/${symbol}`).then(res => res.data),

  getTradingSignals: (): Promise<any> =>
    api.get('/trading-signals').then(res => res.data),

  optimizeAllStrategies: (): Promise<any> =>
    api.post('/optimize-all-strategies').then(res => res.data),

  // Training Info
  getTrainingInfo: (): Promise<any> =>
    api.get('/training-info').then(res => res.data),

  // Health check
  getHealth: (): Promise<any> =>
    api.get('/health').then(res => res.data),
};

export default api;