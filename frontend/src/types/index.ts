export interface Coin {
  id: number;
  symbol: string;
  name: string;
}

export interface MLPrediction {
  model_type: string;
  confidence: number;
  prediction: string;
  created_at: string;
}

export interface Trade {
  id: number;
  coin_symbol: string;
  side: string;
  size: number;
  price: number;
  leverage: number;
  status: string;
  pnl: number;
  opened_at: string;
  closed_at?: string;
  ml_confidence: number;
}

export interface Position {
  symbol: string;
  side: string;
  size: number;
  avgPrice: number;
  markPrice: number;
  unrealisedPnl: number;
  leverage: string;
}

export interface TradingStrategy {
  coin_symbol: string;
  take_profit_percentage: number;
  stop_loss_percentage: number;
  leverage: number;
  position_size_percentage: number;
  confidence_threshold: number;
  updated_by_ai: boolean;
  ai_optimization_reason?: string;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}