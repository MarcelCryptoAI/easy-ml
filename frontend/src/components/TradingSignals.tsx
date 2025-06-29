import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

interface TradingSignal {
  id: string;
  coin_symbol: string;
  signal_type: 'LONG' | 'SHORT' | 'HOLD';
  timestamp: string;
  models_agreed: number;
  total_models: number;
  avg_confidence: number;
  entry_price: number;
  current_price: number;
  position_size_usdt: number;
  status: 'open' | 'closed' | 'cancelled';
  unrealized_pnl_usdt: number;
  unrealized_pnl_percent: number;
  criteria_met: {
    confidence_threshold: boolean;
    model_agreement: boolean;
    risk_management: boolean;
  };
}

export const TradingSignals: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'open' | 'closed'>('all');

  // First fetch all coins and their predictions
  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/coins`);
      if (!response.ok) throw new Error('Failed to fetch coins');
      return response.json();
    },
    refetchInterval: 60000
  });

  // Fetch REAL signals from backend
  const { data: signalsResponse, isLoading, refetch } = useQuery({
    queryKey: ['trading-signals-live'],
    queryFn: async () => {
      console.log('🔍 Fetching LIVE signals from backend...');
      
      try {
        // Use the original /signals endpoint that had 50 signals
        const response = await fetch('https://easy-ml-production.up.railway.app/signals');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch live signals: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Live signals with execution status received:', data);
        
        // Transform backend signals to match our interface
        if (data.signals && Array.isArray(data.signals)) {
          const transformedSignals = data.signals.map((signal: any, index: number) => {
            // Map backend execution status to frontend status
            const statusMap: { [key: string]: 'open' | 'closed' | 'cancelled' } = {
              'pending': 'open',
              'executing': 'open', 
              'executed': 'open',
              'failed': 'cancelled',
              'closed': 'closed'
            };
            
            // Safely extract consensus data
            const consensus = signal.consensus || {};
            const buyCount = consensus.buy_count || 0;
            const sellCount = consensus.sell_count || 0;
            const totalModels = consensus.total_models || 10;
            
            return {
              id: signal.id || `${signal.coin_symbol}_${Date.now()}_${index}`,
              coin_symbol: signal.coin_symbol,
              signal_type: signal.signal_type,
              timestamp: signal.created_at || new Date().toISOString(),
              models_agreed: Math.max(buyCount, sellCount),
              total_models: totalModels,
              avg_confidence: signal.confidence || 0,
              entry_price: signal.entry_price || 0,
              current_price: signal.current_price || signal.entry_price || 0,
              position_size_usdt: signal.position_size_usdt || 1000,
              status: statusMap[signal.execution_status] || 'open',
              unrealized_pnl_usdt: signal.unrealized_pnl_usdt || 0,
              unrealized_pnl_percent: signal.unrealized_pnl_percent || 0,
              criteria_met: {
                confidence_threshold: (signal.confidence || 0) >= 30,
                model_agreement: buyCount >= 2 || sellCount >= 2,
                risk_management: true
              }
            };
          });
          
          return {
            success: true,
            signals: transformedSignals,
            total_signals: transformedSignals.length,
            timestamp: new Date().toISOString()
          };
        }
        
        return {
          success: true,
          signals: [],
          total_signals: 0,
          timestamp: new Date().toISOString()
        };
        
      } catch (error) {
        console.error('❌ Error fetching live signals:', error);
        throw error;
      }
    },
    refetchInterval: 10000, // Refresh every 10 seconds for live data
    retry: 3,
    retryDelay: 1000
  });

  const signals = signalsResponse?.signals || [];

  const filteredSignals = signals.filter((signal: TradingSignal) => {
    const matchesSearch = signal.coin_symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || signal.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getSignalIcon = (type: string) => {
    switch (type) {
      case 'LONG': return <span className="text-green-400 text-xl">📈</span>;
      case 'SHORT': return <span className="text-red-400 text-xl">📉</span>;
      case 'HOLD': return <span className="text-gray-400 text-xl">⏸️</span>;
      default: return <span className="text-gray-400 text-xl">⏸️</span>;
    }
  };

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'LONG': return 'from-green-500/20 to-emerald-500/20 border-green-500/50';
      case 'SHORT': return 'from-red-500/20 to-pink-500/20 border-red-500/50';
      case 'HOLD': return 'from-gray-500/20 to-gray-500/20 border-gray-500/50';
      default: return 'from-gray-500/20 to-gray-500/20 border-gray-500/50';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'from-yellow-500/20 to-orange-500/20 border-yellow-500/50';
      case 'closed': return 'from-green-500/20 to-emerald-500/20 border-green-500/50';
      case 'cancelled': return 'from-red-500/20 to-pink-500/20 border-red-500/50';
      default: return 'from-gray-500/20 to-gray-500/20 border-gray-500/50';
    }
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? 'text-green-400' : 'text-red-400';
  };

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('nl-NL', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const totalPnL = filteredSignals.reduce((sum: number, signal: TradingSignal) => sum + signal.unrealized_pnl_usdt, 0);
  const openPositions = filteredSignals.filter((s: TradingSignal) => s.status === 'open').length;

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500">
            📡 Trading Signals Command Center
          </h1>
          <button
            onClick={() => refetch()}
            className="px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 backdrop-blur-xl border border-cyan-500/50 rounded-xl text-cyan-400 hover:bg-cyan-500/30 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_30px_rgba(0,255,255,0.5)]"
          >
            <span className="text-xl">🔄</span>
          </button>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-cyan-400 mb-2">Total Signals</h3>
              <p className="text-4xl font-bold text-white">{signals.length}</p>
            </div>
          </div>
          
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-yellow-500/30 rounded-2xl p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-2">Open Positions</h3>
              <p className="text-4xl font-bold text-white">{openPositions}</p>
            </div>
          </div>
          
          <div className="relative group">
            <div className={`absolute inset-0 ${totalPnL >= 0 ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20' : 'bg-gradient-to-r from-red-500/20 to-pink-500/20'} rounded-2xl blur-xl`} />
            <div className={`relative bg-black/50 backdrop-blur-xl border ${totalPnL >= 0 ? 'border-green-500/30' : 'border-red-500/30'} rounded-2xl p-6`}>
              <h3 className={`text-xl font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'} mb-2`}>Total P&L (USDT)</h3>
              <p className={`text-4xl font-bold ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
              </p>
            </div>
          </div>
        </div>

        {/* Signal Criteria Alert */}
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-indigo-500/20 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-indigo-600 rounded-xl flex items-center justify-center">
                <span className="text-2xl">📋</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-blue-400 mb-2">Automatic Trading System</h3>
                <div className="text-gray-300 space-y-1">
                  <p>• <strong>Auto Execution:</strong> Signals automatically executed as real trades on ByBit</p>
                  <p>• <strong>Criteria:</strong> 25%+ confidence, 2+ models agree, risk limits enforced</p>
                  <p>• <strong>Safety:</strong> Max $500/trade, $200 daily loss limit, real-time P&L tracking</p>
                  <p>• <strong>Status:</strong> 🚀 LIVE TRADING - Converting your 50+ signals to real positions</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex gap-4 mb-8">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl blur-lg" />
            <div className="relative">
              <input
                type="text"
                placeholder="Search coins..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-80 px-4 py-3 bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-purple-500/60"
              />
              <span className="absolute right-3 top-3 text-gray-400">🔍</span>
            </div>
          </div>
          
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-xl blur-lg" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="relative px-4 py-3 bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-xl text-white focus:outline-none focus:border-cyan-500/60"
            >
              <option value="all">All Status</option>
              <option value="open">Open</option>
              <option value="closed">Closed</option>
            </select>
          </div>
        </div>

        {isLoading ? (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
              <div className="animate-pulse">
                <div className="h-4 bg-purple-500/30 rounded mb-4"></div>
                <div className="h-4 bg-purple-500/20 rounded mb-4"></div>
                <div className="h-4 bg-purple-500/10 rounded"></div>
              </div>
            </div>
          </div>
        ) : (
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-gray-500/10 to-gray-500/10 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-2xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-white">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left p-4 text-cyan-400 font-bold">Date/Time</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Coin</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Signal</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Models</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Confidence</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Entry Price</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Current Price</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Position Size</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">Status</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">P&L (USDT)</th>
                      <th className="text-left p-4 text-cyan-400 font-bold">P&L (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSignals.map((signal: TradingSignal) => (
                      <tr key={signal.id} className="border-b border-gray-800 hover:bg-gradient-to-r hover:from-purple-500/10 hover:to-transparent transition-all duration-300">
                        <td className="p-4 text-gray-300">
                          {formatDateTime(signal.timestamp)}
                        </td>
                        <td className="p-4">
                          <span className="font-bold text-white">{signal.coin_symbol}</span>
                        </td>
                        <td className="p-4">
                          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-gradient-to-r ${getSignalColor(signal.signal_type)} border`}>
                            {getSignalIcon(signal.signal_type)}
                            <span className="font-semibold">{signal.signal_type}</span>
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="inline-flex items-center px-3 py-1 rounded-lg bg-gradient-to-r from-blue-500/20 to-indigo-500/20 border border-blue-500/50">
                            <span className="text-blue-400 font-semibold">{signal.models_agreed}/{signal.total_models}</span>
                          </div>
                        </td>
                        <td className="p-4 text-white">
                          {signal.avg_confidence.toFixed(1)}%
                        </td>
                        <td className="p-4 text-gray-300">
                          ${signal.entry_price.toFixed(4)}
                        </td>
                        <td className="p-4 text-gray-300">
                          ${signal.current_price.toFixed(4)}
                        </td>
                        <td className="p-4 text-gray-300">
                          ${signal.position_size_usdt.toFixed(0)}
                        </td>
                        <td className="p-4">
                          <div className={`inline-flex items-center px-3 py-1 rounded-lg bg-gradient-to-r ${getStatusColor(signal.status)} border`}>
                            <span className="font-semibold">{signal.status.toUpperCase()}</span>
                          </div>
                        </td>
                        <td className={`p-4 font-bold ${getPnLColor(signal.unrealized_pnl_usdt)}`}>
                          {signal.unrealized_pnl_usdt >= 0 ? '+' : ''}
                          {signal.unrealized_pnl_usdt.toFixed(2)}
                        </td>
                        <td className={`p-4 font-bold ${getPnLColor(signal.unrealized_pnl_percent)}`}>
                          {signal.unrealized_pnl_percent >= 0 ? '+' : ''}
                          {signal.unrealized_pnl_percent.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {filteredSignals.length === 0 && !isLoading && (
          <div className="relative mt-8">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-indigo-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="text-3xl">📡</span>
                </div>
                <h3 className="text-xl font-bold text-blue-400 mb-2">No Trading Signals Found</h3>
                <p className="text-gray-300">No signals match your current criteria. The system is actively monitoring for new opportunities.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};