import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  ShowChart,
  Speed,
  EmojiEvents,
  Refresh,
  ArrowUpward,
  ArrowDownward
} from '@mui/icons-material';

// Modern CSS-in-JS styles
const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%)',
    color: '#ffffff',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    overflow: 'hidden',
    position: 'relative' as const,
  },
  
  backgroundOrbs: {
    position: 'absolute' as const,
    inset: 0,
    pointerEvents: 'none' as const,
    zIndex: 0,
  },
  
  orb: {
    position: 'absolute' as const,
    borderRadius: '50%',
    filter: 'blur(60px)',
    opacity: 0.3,
    animation: 'float 6s ease-in-out infinite',
  },
  
  orb1: {
    width: '300px',
    height: '300px',
    background: 'linear-gradient(45deg, #00ffff, #0099ff)',
    top: '10%',
    left: '10%',
    animationDelay: '0s',
  },
  
  orb2: {
    width: '400px',
    height: '400px',
    background: 'linear-gradient(45deg, #ff00ff, #9945ff)',
    bottom: '10%',
    right: '10%',
    animationDelay: '2s',
  },
  
  orb3: {
    width: '350px',
    height: '350px',
    background: 'linear-gradient(45deg, #00ff88, #00cc66)',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    animationDelay: '4s',
  },
  
  content: {
    position: 'relative' as const,
    zIndex: 10,
    padding: '2rem',
  },
  
  header: {
    textAlign: 'center' as const,
    marginBottom: '3rem',
  },
  
  title: {
    fontSize: 'clamp(2.5rem, 8vw, 4rem)',
    fontWeight: 800,
    background: 'linear-gradient(135deg, #00ffff 0%, #ff00ff 50%, #ffff00 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textAlign: 'center' as const,
    marginBottom: '1rem',
    letterSpacing: '-0.02em',
  },
  
  subtitle: {
    fontSize: '1.25rem',
    color: '#94a3b8',
    textAlign: 'center' as const,
    fontWeight: 400,
  },
  
  grid: {
    display: 'grid',
    gap: '2rem',
    marginBottom: '3rem',
  },
  
  grid4: {
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
  },
  
  grid2: {
    gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
  },
  
  grid3: {
    gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
  },
  
  card: {
    background: 'rgba(15, 23, 42, 0.8)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(148, 163, 184, 0.1)',
    borderRadius: '20px',
    padding: '2rem',
    position: 'relative' as const,
    overflow: 'hidden',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
  },
  
  cardHover: {
    transform: 'translateY(-8px)',
    boxShadow: '0 25px 50px rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(148, 163, 184, 0.3)',
  },
  
  cardGlow: {
    position: 'absolute' as const,
    inset: 0,
    background: 'linear-gradient(135deg, transparent 0%, rgba(59, 130, 246, 0.1) 50%, transparent 100%)',
    borderRadius: '20px',
    opacity: 0,
    transition: 'opacity 0.3s ease',
  },
  
  metricCard: {
    position: 'relative' as const,
    overflow: 'hidden',
  },
  
  metricHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '1.5rem',
  },
  
  metricLabel: {
    fontSize: '0.875rem',
    color: '#94a3b8',
    fontWeight: 500,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  },
  
  metricValue: {
    fontSize: '2.5rem',
    fontWeight: 700,
    lineHeight: 1,
    marginTop: '0.5rem',
  },
  
  metricSubtext: {
    fontSize: '0.875rem',
    color: '#64748b',
    marginTop: '0.5rem',
  },
  
  iconContainer: {
    width: '64px',
    height: '64px',
    borderRadius: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '2rem',
  },
  
  progressBar: {
    width: '100%',
    height: '8px',
    backgroundColor: 'rgba(148, 163, 184, 0.2)',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '1rem',
  },
  
  progressFill: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 1s ease-in-out',
  },
  
  chartContainer: {
    height: '400px',
    width: '100%',
    marginTop: '1rem',
  },
  
  chartTitle: {
    fontSize: '1.5rem',
    fontWeight: 600,
    marginBottom: '1rem',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  
  refreshButton: {
    position: 'fixed' as const,
    top: '2rem',
    right: '2rem',
    width: '60px',
    height: '60px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #00ffff, #3b82f6)',
    border: 'none',
    color: '#000',
    fontSize: '1.5rem',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 10px 30px rgba(0, 255, 255, 0.3)',
    transition: 'transform 0.2s ease',
    zIndex: 1000,
  },
  
  refreshButtonHover: {
    transform: 'scale(1.1)',
  },
  
  tradeList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
  },
  
  tradeItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem',
    background: 'rgba(148, 163, 184, 0.05)',
    borderRadius: '12px',
    border: '1px solid rgba(148, 163, 184, 0.1)',
    transition: 'all 0.2s ease',
  },
  
  tradeItemHover: {
    background: 'rgba(148, 163, 184, 0.1)',
    transform: 'translateX(4px)',
  },
  
  tradeLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  
  tradeBadge: {
    width: '40px',
    height: '40px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2rem',
  },
  
  tradeInfo: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.25rem',
  },
  
  tradeCoin: {
    fontSize: '1rem',
    fontWeight: 600,
    color: '#f1f5f9',
  },
  
  tradeSide: {
    fontSize: '0.75rem',
    color: '#94a3b8',
  },
  
  tradeRight: {
    textAlign: 'right' as const,
  },
  
  tradePnl: {
    fontSize: '1.125rem',
    fontWeight: 600,
  },
  
  tradeConfidence: {
    fontSize: '0.75rem',
    color: '#94a3b8',
  },
  
  modelRankings: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '1rem',
  },
  
  modelItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem',
    background: 'rgba(59, 130, 246, 0.05)',
    borderRadius: '12px',
    border: '1px solid rgba(59, 130, 246, 0.2)',
    transition: 'all 0.2s ease',
  },
  
  modelRank: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.25rem',
    fontWeight: 700,
    color: '#000',
  },
  
  modelInfo: {
    flex: 1,
  },
  
  modelName: {
    fontSize: '1.125rem',
    fontWeight: 600,
    color: '#3b82f6',
  },
  
  modelStats: {
    fontSize: '0.875rem',
    color: '#94a3b8',
  },
  
  modelAccuracy: {
    fontSize: '1.5rem',
    fontWeight: 700,
    color: '#3b82f6',
  },
  
  modelRoi: {
    fontSize: '0.875rem',
    fontWeight: 500,
  },
  
  // Color utilities
  positive: { color: '#10b981' },
  negative: { color: '#ef4444' },
  neutral: { color: '#6b7280' },
  cyan: { color: '#06b6d4' },
  purple: { color: '#8b5cf6' },
  orange: { color: '#f59e0b' },
  green: { color: '#10b981' },
  
  // Background utilities
  bgGreen: { background: 'linear-gradient(135deg, #10b981, #059669)' },
  bgCyan: { background: 'linear-gradient(135deg, #06b6d4, #0891b2)' },
  bgPurple: { background: 'linear-gradient(135deg, #8b5cf6, #7c3aed)' },
  bgOrange: { background: 'linear-gradient(135deg, #f59e0b, #d97706)' },
  bgGold: { background: 'linear-gradient(135deg, #fbbf24, #f59e0b)' },
  bgSilver: { background: 'linear-gradient(135deg, #e5e7eb, #9ca3af)' },
  bgBronze: { background: 'linear-gradient(135deg, #d97706, #b45309)' },
};

// Animation keyframes
const globalStyles = `
  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }
  
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

// Interfaces
interface TradingStats {
  total_trades: number;
  open_positions: number;
  total_pnl: number;
  total_volume: number;
  win_rate: number;
  sharp_ratio: number;
  max_drawdown: number;
  roi: number;
  profit_factor: number;
  best_trade: number;
  worst_trade: number;
}

interface ModelPerformance {
  model_type: string;
  accuracy: number;
  total_predictions: number;
  roi_contribution: number;
}

interface RecentTrade {
  coin_symbol: string;
  side: string;
  pnl: number;
  roi: number;
  ml_confidence: number;
}

interface TimeSeriesData {
  timestamp: string;
  balance: number;
  cumulative_pnl: number;
}

interface ModernTradingDashboardProps {
  onNavigate?: (tabIndex: number) => void;
}

export const ModernTradingDashboard: React.FC<ModernTradingDashboardProps> = ({ onNavigate }) => {
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showNavigation, setShowNavigation] = useState(false);

  // Data fetching
  const { data: tradingStats, refetch: refetchStats } = useQuery({
    queryKey: ['trading-statistics'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/statistics`
      );
      if (!response.ok) throw new Error('Failed to fetch trading statistics');
      return response.json();
    },
    refetchInterval: 60000
  });

  const { data: modelPerformance } = useQuery({
    queryKey: ['model-performance'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/models/performance`
      );
      if (!response.ok) throw new Error('Failed to fetch model performance');
      return response.json();
    },
    refetchInterval: 300000
  });

  const { data: recentTrades } = useQuery({
    queryKey: ['recent-trades'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trades/recent?limit=5`
      );
      if (!response.ok) throw new Error('Failed to fetch recent trades');
      return response.json();
    },
    refetchInterval: 30000
  });

  const { data: timeSeriesData } = useQuery({
    queryKey: ['time-series'],
    queryFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/analytics/timeseries?timeframe=24h`
      );
      if (!response.ok) throw new Error('Failed to fetch time series data');
      return response.json();
    },
    refetchInterval: 120000
  });

  // Format helpers
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetchStats();
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  return (
    <>
      <style>{globalStyles}</style>
      <div style={styles.container}>
        {/* Background Orbs */}
        <div style={styles.backgroundOrbs}>
          <div style={{ ...styles.orb, ...styles.orb1 }} />
          <div style={{ ...styles.orb, ...styles.orb2 }} />
          <div style={{ ...styles.orb, ...styles.orb3 }} />
        </div>

        {/* Navigation Menu */}
        {showNavigation && (
          <div style={{
            position: 'fixed',
            top: '50%',
            left: '2rem',
            transform: 'translateY(-50%)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            zIndex: 1000,
          }}>
            {[
              { label: '🎮 Trading', index: 1 },
              { label: '📊 ML Progress', index: 2 },
              { label: '🔧 Training', index: 3 },
              { label: '📡 Signals', index: 4 },
              { label: '⚙️ Strategy', index: 5 },
              { label: '🤖 AI Optimizer', index: 6 },
              { label: '📈 Analysis', index: 7 },
            ].map((item) => (
              <button
                key={item.index}
                onClick={() => onNavigate?.(item.index)}
                style={{
                  padding: '1rem',
                  background: 'rgba(15, 23, 42, 0.9)',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  borderRadius: '12px',
                  color: '#f1f5f9',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  backdropFilter: 'blur(20px)',
                  transition: 'all 0.2s ease',
                  minWidth: '140px',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)';
                  e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(15, 23, 42, 0.9)';
                  e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.3)';
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}

        {/* Navigation Toggle */}
        <button
          style={{
            position: 'fixed',
            top: '2rem',
            left: '2rem',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            border: 'none',
            color: '#fff',
            fontSize: '1.5rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 10px 30px rgba(59, 130, 246, 0.3)',
            transition: 'transform 0.2s ease',
            zIndex: 1000,
          }}
          onClick={() => setShowNavigation(!showNavigation)}
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          ☰
        </button>

        {/* Refresh Button */}
        <button
          style={{
            ...styles.refreshButton,
            ...(isRefreshing ? { transform: 'scale(1.1) rotate(360deg)' } : {})
          }}
          onClick={handleRefresh}
          onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
          onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          <Refresh style={{ transform: isRefreshing ? 'rotate(360deg)' : 'none', transition: 'transform 1s ease' }} />
        </button>

        <div style={styles.content}>
          {/* Header */}
          <div style={styles.header}>
            <h1 style={styles.title}>Neural Trading Command</h1>
            <p style={styles.subtitle}>AI-Powered Real-Time Analytics Dashboard</p>
          </div>

          {/* Key Metrics */}
          <div style={{ ...styles.grid, ...styles.grid4 }}>
            {/* Total PnL */}
            <div
              style={{
                ...styles.card,
                ...styles.metricCard,
                ...(hoveredCard === 'pnl' ? styles.cardHover : {})
              }}
              onMouseEnter={() => setHoveredCard('pnl')}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div style={styles.cardGlow} />
              <div style={styles.metricHeader}>
                <div>
                  <div style={styles.metricLabel}>Total P&L</div>
                  <div style={{
                    ...styles.metricValue,
                    ...(tradingStats?.total_pnl >= 0 ? styles.positive : styles.negative)
                  }}>
                    {formatCurrency(tradingStats?.total_pnl || 0)}
                  </div>
                  <div style={styles.metricSubtext}>
                    ROI: {formatPercentage(tradingStats?.roi || 0)}
                  </div>
                </div>
                <div style={{ ...styles.iconContainer, ...styles.bgGreen }}>
                  <AttachMoney />
                </div>
              </div>
              <div style={styles.progressBar}>
                <div style={{
                  ...styles.progressFill,
                  ...styles.bgGreen,
                  width: `${Math.min(100, Math.abs(tradingStats?.roi || 0) * 2)}%`
                }} />
              </div>
            </div>

            {/* Win Rate */}
            <div
              style={{
                ...styles.card,
                ...styles.metricCard,
                ...(hoveredCard === 'winrate' ? styles.cardHover : {})
              }}
              onMouseEnter={() => setHoveredCard('winrate')}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div style={styles.metricHeader}>
                <div>
                  <div style={styles.metricLabel}>Win Rate</div>
                  <div style={{ ...styles.metricValue, ...styles.cyan }}>
                    {(tradingStats?.win_rate || 0).toFixed(1)}%
                  </div>
                  <div style={styles.metricSubtext}>
                    {tradingStats?.total_trades || 0} total trades
                  </div>
                </div>
                <div style={{ ...styles.iconContainer, ...styles.bgCyan }}>
                  <EmojiEvents />
                </div>
              </div>
              <div style={styles.progressBar}>
                <div style={{
                  ...styles.progressFill,
                  ...styles.bgCyan,
                  width: `${tradingStats?.win_rate || 0}%`
                }} />
              </div>
            </div>

            {/* Open Positions */}
            <div
              style={{
                ...styles.card,
                ...styles.metricCard,
                ...(hoveredCard === 'positions' ? styles.cardHover : {})
              }}
              onMouseEnter={() => setHoveredCard('positions')}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div style={styles.metricHeader}>
                <div>
                  <div style={styles.metricLabel}>Open Positions</div>
                  <div style={{ ...styles.metricValue, ...styles.purple }}>
                    {tradingStats?.open_positions || 0}
                  </div>
                  <div style={styles.metricSubtext}>
                    Volume: {formatCurrency(tradingStats?.total_volume || 0)}
                  </div>
                </div>
                <div style={{ ...styles.iconContainer, ...styles.bgPurple }}>
                  <ShowChart />
                </div>
              </div>
            </div>

            {/* Sharpe Ratio */}
            <div
              style={{
                ...styles.card,
                ...styles.metricCard,
                ...(hoveredCard === 'sharpe' ? styles.cardHover : {})
              }}
              onMouseEnter={() => setHoveredCard('sharpe')}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div style={styles.metricHeader}>
                <div>
                  <div style={styles.metricLabel}>Sharpe Ratio</div>
                  <div style={{ ...styles.metricValue, ...styles.orange }}>
                    {(tradingStats?.sharp_ratio || 0).toFixed(2)}
                  </div>
                  <div style={styles.metricSubtext}>
                    Max DD: {formatPercentage(tradingStats?.max_drawdown || 0)}
                  </div>
                </div>
                <div style={{ ...styles.iconContainer, ...styles.bgOrange }}>
                  <Speed />
                </div>
              </div>
            </div>
          </div>

          {/* Charts Section */}
          <div style={{ ...styles.grid, ...styles.grid2 }}>
            {/* Portfolio Performance */}
            <div style={styles.card}>
              <h3 style={styles.chartTitle}>Portfolio Performance</h3>
              <div style={styles.chartContainer}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={timeSeriesData || []}>
                    <defs>
                      <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.1}/>
                      </linearGradient>
                      <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => format(new Date(value), 'HH:mm')}
                      stroke="#94a3b8"
                    />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: '12px',
                        backdropFilter: 'blur(20px)'
                      }}
                      formatter={(value: number) => formatCurrency(value)}
                    />
                    <Area
                      type="monotone"
                      dataKey="balance"
                      stroke="#06b6d4"
                      fill="url(#balanceGradient)"
                      strokeWidth={3}
                      name="Balance"
                    />
                    <Area
                      type="monotone"
                      dataKey="cumulative_pnl"
                      stroke="#8b5cf6"
                      fill="url(#pnlGradient)"
                      strokeWidth={3}
                      name="Cumulative PnL"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Model Performance Radar */}
            <div style={styles.card}>
              <h3 style={styles.chartTitle}>AI Model Performance</h3>
              <div style={styles.chartContainer}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={modelPerformance?.slice(0, 6) || []}>
                    <PolarGrid stroke="rgba(148, 163, 184, 0.2)" />
                    <PolarAngleAxis dataKey="model_type" stroke="#94a3b8" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#94a3b8" />
                    <Radar
                      name="Accuracy"
                      dataKey="accuracy"
                      stroke="#06b6d4"
                      fill="#06b6d4"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Radar
                      name="ROI Impact"
                      dataKey="roi_contribution"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Bottom Section */}
          <div style={{ ...styles.grid, ...styles.grid2 }}>
            {/* Top Models */}
            <div style={styles.card}>
              <h3 style={styles.chartTitle}>Neural Network Rankings</h3>
              <div style={styles.modelRankings}>
                {modelPerformance?.slice(0, 5).map((model: ModelPerformance, index: number) => (
                  <div key={model.model_type} style={styles.modelItem}>
                    <div style={{
                      ...styles.modelRank,
                      ...(index === 0 ? styles.bgGold : 
                          index === 1 ? styles.bgSilver : 
                          index === 2 ? styles.bgBronze : styles.bgCyan)
                    }}>
                      {index + 1}
                    </div>
                    <div style={styles.modelInfo}>
                      <div style={styles.modelName}>{model.model_type.toUpperCase()}</div>
                      <div style={styles.modelStats}>{model.total_predictions.toLocaleString()} predictions</div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={styles.modelAccuracy}>{model.accuracy.toFixed(1)}%</div>
                      <div style={{
                        ...styles.modelRoi,
                        ...(model.roi_contribution >= 0 ? styles.positive : styles.negative)
                      }}>
                        ROI: {formatPercentage(model.roi_contribution)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Trades */}
            <div style={styles.card}>
              <h3 style={styles.chartTitle}>Live Trading Activity</h3>
              <div style={styles.tradeList}>
                {recentTrades?.map((trade: RecentTrade, index: number) => (
                  <div key={index} style={styles.tradeItem}>
                    <div style={styles.tradeLeft}>
                      <div style={{
                        ...styles.tradeBadge,
                        ...(trade.side === 'LONG' ? 
                          { background: 'rgba(16, 185, 129, 0.2)', border: '1px solid #10b981' } : 
                          { background: 'rgba(239, 68, 68, 0.2)', border: '1px solid #ef4444' })
                      }}>
                        {trade.side === 'LONG' ? <ArrowUpward style={styles.positive} /> : <ArrowDownward style={styles.negative} />}
                      </div>
                      <div style={styles.tradeInfo}>
                        <div style={styles.tradeCoin}>{trade.coin_symbol}</div>
                        <div style={styles.tradeSide}>{trade.side}</div>
                      </div>
                    </div>
                    <div style={styles.tradeRight}>
                      <div style={{
                        ...styles.tradePnl,
                        ...(trade.pnl >= 0 ? styles.positive : styles.negative)
                      }}>
                        {formatCurrency(trade.pnl)}
                      </div>
                      <div style={styles.tradeConfidence}>ML: {trade.ml_confidence}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Risk Analytics */}
          <div style={{ ...styles.grid, ...styles.grid3 }}>
            <div style={styles.card}>
              <h3 style={{ ...styles.chartTitle, background: 'linear-gradient(135deg, #ef4444, #f59e0b)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Risk Management
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Max Drawdown</span>
                  <span style={{ ...styles.negative, fontWeight: 600 }}>{formatPercentage(tradingStats?.max_drawdown || 0)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Profit Factor</span>
                  <span style={{ ...styles.positive, fontWeight: 600 }}>{(tradingStats?.profit_factor || 0).toFixed(2)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Best Trade</span>
                  <span style={{ ...styles.positive, fontWeight: 600 }}>{formatCurrency(tradingStats?.best_trade || 0)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Worst Trade</span>
                  <span style={{ ...styles.negative, fontWeight: 600 }}>{formatCurrency(tradingStats?.worst_trade || 0)}</span>
                </div>
              </div>
            </div>

            <div style={styles.card}>
              <h3 style={{ ...styles.chartTitle, background: 'linear-gradient(135deg, #10b981, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                System Status
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Active Strategies</span>
                  <span style={{ ...styles.green, fontWeight: 600 }}>512</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Models Running</span>
                  <span style={{ ...styles.green, fontWeight: 600 }}>10</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Predictions/Hour</span>
                  <span style={{ ...styles.green, fontWeight: 600 }}>~850</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Status</span>
                  <span style={{ ...styles.green, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '8px', height: '8px', background: '#10b981', borderRadius: '50%', animation: 'pulse 2s infinite' }} />
                    ONLINE
                  </span>
                </div>
              </div>
            </div>

            <div style={styles.card}>
              <h3 style={{ ...styles.chartTitle, background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                24H Performance
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Daily PnL</span>
                  <span style={{ ...styles.positive, fontWeight: 600 }}>+$234.56</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Trades Today</span>
                  <span style={{ ...styles.purple, fontWeight: 600 }}>23</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Win Rate 24H</span>
                  <span style={{ ...styles.positive, fontWeight: 600 }}>78.3%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#94a3b8' }}>Volume 24H</span>
                  <span style={{ ...styles.purple, fontWeight: 600 }}>$12.4K</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};