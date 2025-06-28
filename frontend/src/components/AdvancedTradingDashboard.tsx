import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Speed,
  EmojiEvents,
  AttachMoney,
  Timeline,
  Assessment,
  Refresh,
  ArrowUpward,
  ArrowDownward,
  AutoGraph,
  QueryStats,
  Analytics,
  CandlestickChart
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';

// Define interfaces for our data
interface TradingStats {
  total_trades: number;
  open_positions: number;
  closed_positions: number;
  total_pnl: number;
  total_volume: number;
  win_rate: number;
  avg_profit: number;
  avg_loss: number;
  best_trade: number;
  worst_trade: number;
  sharp_ratio: number;
  max_drawdown: number;
  roi: number;
  profit_factor: number;
}

interface ModelPerformance {
  model_type: string;
  accuracy: number;
  total_predictions: number;
  successful_predictions: number;
  avg_confidence: number;
  roi_contribution: number;
}

interface RecentTrade {
  coin_symbol: string;
  side: string;
  pnl: number;
  roi: number;
  opened_at: string;
  closed_at: string;
  ml_confidence: number;
}

interface TimeSeriesData {
  timestamp: string;
  balance: number;
  cumulative_pnl: number;
  trades_count: number;
}

export const AdvancedTradingDashboard: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  // Fetch overall trading statistics
  const { data: tradingStats, refetch: refetchStats } = useQuery({
    queryKey: ['trading-statistics'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/statistics`
      );
      if (!response.ok) throw new Error('Failed to fetch trading statistics');
      return response.json();
    },
    refetchInterval: autoRefresh ? 60000 : false
  });

  // Fetch model performance data
  const { data: modelPerformance } = useQuery({
    queryKey: ['model-performance'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/models/performance`
      );
      if (!response.ok) throw new Error('Failed to fetch model performance');
      return response.json();
    },
    refetchInterval: autoRefresh ? 300000 : false
  });

  // Fetch recent trades
  const { data: recentTrades } = useQuery({
    queryKey: ['recent-trades'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trades/recent?limit=10`
      );
      if (!response.ok) throw new Error('Failed to fetch recent trades');
      return response.json();
    },
    refetchInterval: autoRefresh ? 30000 : false
  });

  // Fetch time series data for charts
  const { data: timeSeriesData } = useQuery({
    queryKey: ['time-series', selectedTimeframe],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/analytics/timeseries?timeframe=${selectedTimeframe}`
      );
      if (!response.ok) throw new Error('Failed to fetch time series data');
      return response.json();
    },
    refetchInterval: autoRefresh ? 120000 : false
  });

  // Fetch PnL distribution
  const { data: pnlDistribution } = useQuery({
    queryKey: ['pnl-distribution'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/analytics/pnl-distribution`
      );
      if (!response.ok) throw new Error('Failed to fetch PnL distribution');
      return response.json();
    },
    refetchInterval: false
  });

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Neon color palette
  const neonColors = {
    cyan: '#00ffff',
    magenta: '#ff00ff',
    yellow: '#ffff00',
    green: '#00ff00',
    blue: '#0099ff',
    purple: '#9945ff',
    orange: '#ff6600',
    pink: '#ff0099'
  };

  return (
    <div className="min-h-screen bg-black overflow-hidden relative">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-2000" />
          <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse animation-delay-4000" />
        </div>
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <div className="mb-8 flex justify-between items-center">
          <div>
            <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 mb-2">
              Trading Neural Command Center
            </h1>
            <p className="text-gray-400 text-lg">Real-time AI-Powered Trading Analytics</p>
          </div>
          <div className="flex gap-4">
            <button
              onClick={() => refetchStats()}
              className="px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 backdrop-blur-xl border border-cyan-500/50 rounded-xl text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_30px_rgba(0,255,255,0.5)]"
            >
              <Refresh className="animate-spin-slow" />
            </button>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total PnL Card */}
          <div
            onMouseEnter={() => setHoveredCard('pnl')}
            onMouseLeave={() => setHoveredCard(null)}
            className={`relative group ${hoveredCard === 'pnl' ? 'z-20' : ''}`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6 hover:transform hover:scale-105 transition-all duration-300 hover:shadow-[0_20px_50px_rgba(0,255,0,0.5)]">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Total Profit/Loss</p>
                  <p className={`text-4xl font-bold ${(tradingStats?.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatCurrency(tradingStats?.total_pnl || 0)}
                  </p>
                </div>
                <div className="w-16 h-16 bg-gradient-to-br from-green-400 to-emerald-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(0,255,0,0.5)]">
                  <AttachMoney className="text-black text-3xl" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">ROI</span>
                <span className={`text-lg font-semibold ${(tradingStats?.roi || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercentage(tradingStats?.roi || 0)}
                </span>
              </div>
              <div className="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-green-400 to-emerald-400 rounded-full transition-all duration-1000"
                  style={{ width: `${Math.min(100, Math.abs(tradingStats?.roi || 0))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Win Rate Card */}
          <div
            onMouseEnter={() => setHoveredCard('winrate')}
            onMouseLeave={() => setHoveredCard(null)}
            className={`relative group ${hoveredCard === 'winrate' ? 'z-20' : ''}`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-2xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-6 hover:transform hover:scale-105 transition-all duration-300 hover:shadow-[0_20px_50px_rgba(0,255,255,0.5)]">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Win Rate</p>
                  <p className="text-4xl font-bold text-cyan-400">
                    {(tradingStats?.win_rate || 0).toFixed(1)}%
                  </p>
                </div>
                <div className="w-16 h-16 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(0,255,255,0.5)]">
                  <EmojiEvents className="text-black text-3xl" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Total Trades</span>
                <span className="text-lg font-semibold text-cyan-400">
                  {tradingStats?.total_trades || 0}
                </span>
              </div>
              <div className="mt-4 relative h-24">
                <div className="absolute inset-0 flex items-end justify-between">
                  {[...Array(10)].map((_, i) => (
                    <div
                      key={i}
                      className="w-6 bg-gradient-to-t from-cyan-500 to-cyan-300 rounded-t"
                      style={{ 
                        height: `${Math.random() * 100}%`,
                        opacity: 0.7 + (i * 0.03)
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Open Positions Card */}
          <div
            onMouseEnter={() => setHoveredCard('positions')}
            onMouseLeave={() => setHoveredCard(null)}
            className={`relative group ${hoveredCard === 'positions' ? 'z-20' : ''}`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6 hover:transform hover:scale-105 transition-all duration-300 hover:shadow-[0_20px_50px_rgba(147,51,234,0.5)]">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Open Positions</p>
                  <p className="text-4xl font-bold text-purple-400">
                    {tradingStats?.open_positions || 0}
                  </p>
                </div>
                <div className="w-16 h-16 bg-gradient-to-br from-purple-400 to-pink-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(147,51,234,0.5)]">
                  <ShowChart className="text-black text-3xl" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Volume</span>
                <span className="text-lg font-semibold text-purple-400">
                  {formatCurrency(tradingStats?.total_volume || 0)}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-1">
                {[...Array(9)].map((_, i) => (
                  <div
                    key={i}
                    className="h-8 bg-gradient-to-br from-purple-500/50 to-pink-500/50 rounded animate-pulse"
                    style={{ animationDelay: `${i * 100}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Sharpe Ratio Card */}
          <div
            onMouseEnter={() => setHoveredCard('sharpe')}
            onMouseLeave={() => setHoveredCard(null)}
            className={`relative group ${hoveredCard === 'sharpe' ? 'z-20' : ''}`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-2xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-300" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6 hover:transform hover:scale-105 transition-all duration-300 hover:shadow-[0_20px_50px_rgba(251,146,60,0.5)]">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Sharpe Ratio</p>
                  <p className="text-4xl font-bold text-orange-400">
                    {(tradingStats?.sharp_ratio || 0).toFixed(2)}
                  </p>
                </div>
                <div className="w-16 h-16 bg-gradient-to-br from-orange-400 to-yellow-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(251,146,60,0.5)]">
                  <Speed className="text-black text-3xl" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500 text-sm">Max Drawdown</span>
                <span className="text-lg font-semibold text-orange-400">
                  {formatPercentage(tradingStats?.max_drawdown || 0)}
                </span>
              </div>
              <div className="mt-4 relative">
                <svg className="w-full h-16" viewBox="0 0 100 40">
                  <path
                    d="M0,20 Q25,5 50,20 T100,20"
                    stroke="url(#gradient)"
                    strokeWidth="3"
                    fill="none"
                    className="animate-pulse"
                  />
                  <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#fb923c" />
                      <stop offset="100%" stopColor="#fbbf24" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Main Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Portfolio Performance Chart */}
          <div className="lg:col-span-2">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
                <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-6">
                  Portfolio Performance
                </h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeriesData || []}>
                      <defs>
                        <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00ffff" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#00ffff" stopOpacity={0.1}/>
                        </linearGradient>
                        <linearGradient id="colorPnL" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ff00ff" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#ff00ff" stopOpacity={0.1}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(value) => format(new Date(value), 'MMM dd')}
                        stroke="#666"
                      />
                      <YAxis stroke="#666" />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: 'rgba(0,0,0,0.9)',
                          border: '1px solid #333',
                          borderRadius: '12px',
                          backdropFilter: 'blur(10px)'
                        }}
                        formatter={(value: number) => formatCurrency(value)}
                      />
                      <Area
                        type="monotone"
                        dataKey="balance"
                        stroke="#00ffff"
                        fill="url(#colorBalance)"
                        strokeWidth={3}
                        name="Balance"
                      />
                      <Area
                        type="monotone"
                        dataKey="cumulative_pnl"
                        stroke="#ff00ff"
                        fill="url(#colorPnL)"
                        strokeWidth={3}
                        name="Cumulative PnL"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Model Performance Radar */}
          <div>
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
                <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-6">
                  AI Model Performance
                </h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={modelPerformance || []}>
                      <PolarGrid stroke="#333" />
                      <PolarAngleAxis dataKey="model_type" stroke="#666" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#666" />
                      <Radar
                        name="Accuracy"
                        dataKey="accuracy"
                        stroke="#00ffff"
                        fill="#00ffff"
                        fillOpacity={0.3}
                        strokeWidth={2}
                      />
                      <Radar
                        name="ROI Impact"
                        dataKey="roi_contribution"
                        stroke="#ff00ff"
                        fill="#ff00ff"
                        fillOpacity={0.3}
                        strokeWidth={2}
                      />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top ML Models */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400 mb-6">
                Neural Network Rankings
              </h3>
              <div className="space-y-4">
                {modelPerformance?.slice(0, 5).map((model: ModelPerformance, index: number) => (
                  <div key={model.model_type} className="relative">
                    <div className="flex items-center justify-between p-4 bg-gradient-to-r from-cyan-500/10 to-transparent rounded-xl border border-cyan-500/20 hover:border-cyan-500/50 transition-all duration-300">
                      <div className="flex items-center gap-4">
                        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${
                          index === 0 ? 'from-yellow-400 to-orange-500' :
                          index === 1 ? 'from-gray-300 to-gray-500' :
                          index === 2 ? 'from-orange-400 to-orange-600' :
                          'from-cyan-400 to-blue-500'
                        } flex items-center justify-center font-bold text-black text-xl shadow-[0_0_20px_rgba(0,255,255,0.5)]`}>
                          {index + 1}
                        </div>
                        <div>
                          <p className="text-cyan-400 font-semibold text-lg">{model.model_type.toUpperCase()}</p>
                          <p className="text-gray-500 text-sm">{model.total_predictions.toLocaleString()} predictions</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold text-cyan-400">{model.accuracy.toFixed(1)}%</p>
                        <p className={`text-sm ${model.roi_contribution >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ROI: {formatPercentage(model.roi_contribution)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent Trades */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-6">
                Live Trading Activity
              </h3>
              <div className="space-y-3">
                {recentTrades?.slice(0, 5).map((trade: RecentTrade, index: number) => (
                  <div key={index} className="relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 to-transparent animate-pulse" />
                    <div className="relative flex items-center justify-between p-4 bg-gradient-to-r from-purple-500/10 to-transparent rounded-xl border border-purple-500/20 hover:border-purple-500/50 transition-all duration-300">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-lg ${
                          trade.side === 'LONG' ? 'bg-green-500/20 border border-green-500' : 'bg-red-500/20 border border-red-500'
                        } flex items-center justify-center`}>
                          {trade.side === 'LONG' ? <ArrowUpward className="text-green-400" /> : <ArrowDownward className="text-red-400" />}
                        </div>
                        <div>
                          <p className="text-purple-400 font-semibold">{trade.coin_symbol}</p>
                          <p className="text-gray-500 text-sm">{trade.side}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-lg font-bold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatCurrency(trade.pnl)}
                        </p>
                        <p className="text-gray-500 text-sm">
                          ML: {trade.ml_confidence}%
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes animate-pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .animate-spin-slow {
          animation: spin 3s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};