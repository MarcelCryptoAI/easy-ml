import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import { Coin, MLPrediction, TradingStrategy } from '../types';
import toast from 'react-hot-toast';

interface ManualTradeData {
  amount_percentage: number;
  order_type: 'market' | 'limit';
  limit_price?: number;
  leverage: number;
  margin_mode: 'cross' | 'isolated';
  take_profit_percentage: number;
  stop_loss_percentage: number;
  side: 'buy' | 'sell';
}

export const CoinAnalysis: React.FC = () => {
  const [selectedCoin, setSelectedCoin] = useState<string>('');
  const [optimizing, setOptimizing] = useState(false);
  const [priorityTraining, setPriorityTraining] = useState(false);
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [availableBalance, setAvailableBalance] = useState(0);
  const [tradeData, setTradeData] = useState<ManualTradeData>({
    amount_percentage: 5,
    order_type: 'market',
    leverage: 10,
    margin_mode: 'cross',
    take_profit_percentage: 2.5,
    stop_loss_percentage: 1.5,
    side: 'buy'
  });
  const queryClient = useQueryClient();

  const { data: coins = [] } = useQuery({
    queryKey: ['coins'],
    queryFn: tradingApi.getCoins
  });

  const { data: predictions = [], isLoading: predictionsLoading } = useQuery({
    queryKey: ['predictions', selectedCoin],
    queryFn: () => tradingApi.getPredictions(selectedCoin),
    enabled: !!selectedCoin,
    refetchInterval: 30000
  });

  const { data: strategy, refetch: refetchStrategy } = useQuery({
    queryKey: ['strategy', selectedCoin],
    queryFn: () => tradingApi.getStrategy(selectedCoin),
    enabled: !!selectedCoin
  });

  // Priority training mutation
  const priorityTrainingMutation = useMutation({
    mutationFn: async (coinSymbol: string) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/priority`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coin_symbol: coinSymbol })
      });
      if (!response.ok) throw new Error('Failed to start priority training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('🚀 Priority training started for all 10 models!');
      setPriorityTraining(true);
      // Auto-refresh predictions
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['predictions', selectedCoin] });
      }, 2000);
    },
    onError: () => {
      toast.error('Failed to start priority training');
    }
  });

  // Manual trade mutation
  const manualTradeMutation = useMutation({
    mutationFn: async (trade: ManualTradeData & { coin_symbol: string, current_price: number }) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trade)
      });
      if (!response.ok) throw new Error('Failed to execute manual trade');
      return response.json();
    },
    onSuccess: (data) => {
      toast.success(`✅ ${tradeData.side.toUpperCase()} order executed!`);
      setConfirmDialogOpen(false);
      setTradeDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: (error) => {
      toast.error(`❌ Trade failed: ${error.message}`);
    }
  });

  const handleOptimizeStrategy = async () => {
    if (!selectedCoin) return;
    
    setOptimizing(true);
    try {
      const result = await tradingApi.optimizeStrategy(selectedCoin);
      if (result.success) {
        toast.success('Strategy optimized successfully');
        refetchStrategy();
      } else {
        toast.error(result.error || 'Optimization failed');
      }
    } catch (error) {
      toast.error('Failed to optimize strategy');
    } finally {
      setOptimizing(false);
    }
  };

  const handlePriorityTraining = () => {
    if (!selectedCoin) return;
    priorityTrainingMutation.mutate(selectedCoin);
  };

  const handleOpenTradeDialog = async (side: 'buy' | 'sell') => {
    if (!selectedCoin) return;
    
    // Get current price and balance
    try {
      const [priceResponse, balanceResponse] = await Promise.all([
        fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/price/${selectedCoin}`),
        fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/status`)
      ]);
      
      const priceData = await priceResponse.json();
      const balanceData = await balanceResponse.json();
      
      setCurrentPrice(priceData.price || 0);
      setAvailableBalance(balanceData.balance || 0);
      setTradeData(prev => ({ ...prev, side }));
      setTradeDialogOpen(true);
    } catch (error) {
      toast.error('Failed to get current price and balance');
    }
  };

  const handleConfirmTrade = () => {
    setConfirmDialogOpen(true);
  };

  const handleExecuteTrade = () => {
    if (!selectedCoin) return;
    
    manualTradeMutation.mutate({
      ...tradeData,
      coin_symbol: selectedCoin,
      current_price: currentPrice
    });
  };

  // Calculate leveraged percentages for display
  const calculateLeverageDisplay = (percentage: number, leverage: number, isProfit: boolean) => {
    const leveragedPercentage = percentage * leverage;
    const sign = isProfit ? '+' : '-';
    return `${percentage}% (${sign}${leveragedPercentage}%)`;
  };

  // Calculate trade amounts
  const calculateTradeAmounts = () => {
    const tradeAmount = (availableBalance * tradeData.amount_percentage) / 100;
    const positionSize = tradeAmount / currentPrice;
    return { tradeAmount, positionSize };
  };

  const getSignalColor = (prediction: string) => {
    switch (prediction) {
      case 'buy': return 'success';
      case 'sell': return 'error';
      default: return 'default';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  // ALLEEN echte ML data - geen fallback waardes
  const averageConfidence = predictions?.length > 0 
    ? predictions.reduce((sum, pred) => sum + pred.confidence, 0) / predictions.length
    : null;

  const buySignals = predictions?.filter(p => p.prediction === 'buy').length || 0;
  const sellSignals = predictions?.filter(p => p.prediction === 'sell').length || 0;
  const holdSignals = predictions?.filter(p => p.prediction === 'hold').length || 0;

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-900/20 via-black to-purple-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-500 to-pink-500 mb-8">
          🔍 Coin Analysis Hub
        </h1>

        {/* Coin Selection */}
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-indigo-500/30 rounded-2xl p-6">
            <div className="mb-6">
              <label className="block text-gray-300 text-sm mb-2">Select Coin for Analysis</label>
              <select
                value={selectedCoin}
                onChange={(e) => setSelectedCoin(e.target.value)}
                className="w-full px-4 py-3 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
              >
                <option value="">Choose a coin...</option>
                {coins.map((coin: Coin) => (
                  <option key={coin.id} value={coin.symbol}>
                    {coin.symbol} - {coin.name}
                  </option>
                ))}
              </select>
            </div>
            
            {/* Action Buttons */}
            {selectedCoin && (
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={handlePriorityTraining}
                  disabled={priorityTrainingMutation.isPending || priorityTraining}
                  className="px-6 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300 disabled:opacity-50"
                >
                  <div className="flex items-center gap-2">
                    <span>⚡</span>
                    <span>{priorityTraining ? 'Training in Progress...' : 'Priority Train All 10 Models'}</span>
                  </div>
                </button>
                
                <button
                  onClick={() => handleOpenTradeDialog('buy')}
                  className="px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 backdrop-blur-xl border border-green-500/50 rounded-xl text-green-400 hover:bg-green-500/30 transition-all duration-300 hover:scale-105"
                >
                  <div className="flex items-center gap-2">
                    <span>📈</span>
                    <span>Manual LONG</span>
                  </div>
                </button>
                
                <button
                  onClick={() => handleOpenTradeDialog('sell')}
                  className="px-6 py-3 bg-gradient-to-r from-red-500/20 to-pink-500/20 backdrop-blur-xl border border-red-500/50 rounded-xl text-red-400 hover:bg-red-500/30 transition-all duration-300 hover:scale-105"
                >
                  <div className="flex items-center gap-2">
                    <span>📉</span>
                    <span>Manual SHORT</span>
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>

        {selectedCoin && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* ML Predictions Summary */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-blue-400">
                    ML Predictions for {selectedCoin}
                  </h3>
                  {predictionsLoading && (
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-400"></div>
                  )}
                </div>

                {/* Signal Count Cards */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="text-center">
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-xl blur-lg" />
                      <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-xl p-4">
                        <p className="text-gray-400 text-sm mb-1">Buy Signals</p>
                        <p className="text-3xl font-bold text-green-400">{buySignals}</p>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-gray-500/20 to-gray-500/20 rounded-xl blur-lg" />
                      <div className="relative bg-black/50 backdrop-blur-xl border border-gray-500/30 rounded-xl p-4">
                        <p className="text-gray-400 text-sm mb-1">Hold Signals</p>
                        <p className="text-3xl font-bold text-white">{holdSignals}</p>
                      </div>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-pink-500/20 rounded-xl blur-lg" />
                      <div className="relative bg-black/50 backdrop-blur-xl border border-red-500/30 rounded-xl p-4">
                        <p className="text-gray-400 text-sm mb-1">Sell Signals</p>
                        <p className="text-3xl font-bold text-red-400">{sellSignals}</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Average Confidence */}
                <div className="relative mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-xl blur-lg" />
                  <div className="relative bg-black/50 backdrop-blur-xl border border-yellow-500/30 rounded-xl p-4">
                    <p className="text-gray-400 text-sm mb-2">Average Confidence</p>
                    {averageConfidence !== null ? (
                      <div className={`inline-flex items-center px-3 py-1 rounded-lg ${
                        averageConfidence >= 80 ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' :
                        averageConfidence >= 60 ? 'bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/50' :
                        'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50'
                      }`}>
                        <span className={`font-bold text-lg ${
                          averageConfidence >= 80 ? 'text-green-400' :
                          averageConfidence >= 60 ? 'text-yellow-400' :
                          'text-red-400'
                        }`}>
                          {averageConfidence.toFixed(1)}%
                        </span>
                      </div>
                    ) : (
                      <p className="text-gray-400">Waiting for ML predictions...</p>
                    )}
                  </div>
                </div>

                {/* Individual Predictions Table */}
                {predictions?.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-white">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left p-3 text-cyan-400 font-bold">Model</th>
                          <th className="text-left p-3 text-cyan-400 font-bold">Prediction</th>
                          <th className="text-left p-3 text-cyan-400 font-bold">Confidence</th>
                          <th className="text-left p-3 text-cyan-400 font-bold">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictions.map((prediction: MLPrediction) => (
                          <tr key={prediction.model_type} className="border-b border-gray-800 hover:bg-gradient-to-r hover:from-blue-500/10 hover:to-transparent transition-all duration-300">
                            <td className="p-3 font-semibold">{prediction.model_type}</td>
                            <td className="p-3">
                              <div className={`inline-flex items-center px-2 py-1 rounded-lg text-sm ${
                                prediction.prediction === 'buy' ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 text-green-400' :
                                prediction.prediction === 'sell' ? 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50 text-red-400' :
                                'bg-gradient-to-r from-gray-500/20 to-gray-500/20 border border-gray-500/50 text-gray-400'
                              }`}>
                                {prediction.prediction}
                              </div>
                            </td>
                            <td className="p-3">
                              <div className={`inline-flex items-center px-2 py-1 rounded-lg text-sm ${
                                prediction.confidence >= 80 ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50 text-green-400' :
                                prediction.confidence >= 60 ? 'bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/50 text-yellow-400' :
                                'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50 text-red-400'
                              }`}>
                                {prediction.confidence.toFixed(1)}%
                              </div>
                            </td>
                            <td className="p-3 text-gray-400 text-sm">
                              {new Date(prediction.created_at).toLocaleTimeString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4">
                      <span className="text-3xl">🤖</span>
                    </div>
                    <h3 className="text-xl font-bold text-blue-400 mb-2">ML Models Training</h3>
                    <p className="text-gray-300">Predictions will appear here once training is complete</p>
                  </div>
                )}
              </div>
            </div>

            {/* Trading Strategy */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-2xl p-6">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-orange-400">Trading Strategy</h3>
                  <button
                    onClick={handleOptimizeStrategy}
                    disabled={optimizing}
                    className="px-4 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300 disabled:opacity-50"
                  >
                    <div className="flex items-center gap-2">
                      {optimizing ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-400"></div>
                      ) : (
                        <span>🤖</span>
                      )}
                      <span>AI Optimize</span>
                    </div>
                  </button>
                </div>

                {strategy && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-xl blur-lg" />
                        <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-xl p-4">
                          <p className="text-gray-400 text-sm mb-1">Take Profit</p>
                          <p className="text-xl font-bold text-green-400">{strategy.take_profit_percentage}%</p>
                        </div>
                      </div>
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-pink-500/20 rounded-xl blur-lg" />
                        <div className="relative bg-black/50 backdrop-blur-xl border border-red-500/30 rounded-xl p-4">
                          <p className="text-gray-400 text-sm mb-1">Stop Loss</p>
                          <p className="text-xl font-bold text-red-400">{strategy.stop_loss_percentage}%</p>
                        </div>
                      </div>
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-xl blur-lg" />
                        <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-xl p-4">
                          <p className="text-gray-400 text-sm mb-1">Leverage</p>
                          <p className="text-xl font-bold text-blue-400">{strategy.leverage}x</p>
                        </div>
                      </div>
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl blur-lg" />
                        <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-xl p-4">
                          <p className="text-gray-400 text-sm mb-1">Confidence Threshold</p>
                          <p className="text-xl font-bold text-purple-400">{strategy.confidence_threshold}%</p>
                        </div>
                      </div>
                    </div>
                    
                    {strategy.updated_by_ai && (
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-xl blur-lg" />
                        <div className="relative bg-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-xl p-4">
                          <h4 className="text-lg font-bold text-cyan-400 mb-2">🤖 AI Optimization Applied</h4>
                          <p className="text-gray-300">{strategy.ai_optimization_reason}</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Manual Trading Dialog */}
        {tradeDialogOpen && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="relative max-w-4xl w-full max-h-[90vh] overflow-y-auto">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/90 backdrop-blur-xl border border-indigo-500/30 rounded-2xl p-6">
                <h3 className="text-2xl font-bold text-indigo-400 mb-6">
                  Manual {tradeData.side === 'buy' ? 'LONG' : 'SHORT'} Order - {selectedCoin}
                </h3>
                
                {/* Current Price & Balance */}
                <div className="relative mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-xl blur-lg" />
                  <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-xl p-4">
                    <p className="text-blue-400">
                      <strong>Current Price:</strong> ${currentPrice.toFixed(4)} | 
                      <strong> Available Balance:</strong> {availableBalance.toFixed(2)} USDT
                    </p>
                  </div>
                </div>
                
                {/* Trade Form */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Amount (% of balance)</label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      step="1"
                      value={tradeData.amount_percentage}
                      onChange={(e) => setTradeData(prev => ({ ...prev, amount_percentage: Number(e.target.value) }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">≈ {calculateTradeAmounts().tradeAmount.toFixed(2)} USDT</p>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Order Type</label>
                    <select
                      value={tradeData.order_type}
                      onChange={(e) => setTradeData(prev => ({ ...prev, order_type: e.target.value as 'market' | 'limit' }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    >
                      <option value="market">Market Order</option>
                      <option value="limit">Limit Order</option>
                    </select>
                  </div>

                  {tradeData.order_type === 'limit' && (
                    <div>
                      <label className="block text-gray-300 text-sm mb-2">Limit Price (USDT)</label>
                      <input
                        type="number"
                        min="0"
                        step="0.0001"
                        value={tradeData.limit_price || currentPrice}
                        onChange={(e) => setTradeData(prev => ({ ...prev, limit_price: Number(e.target.value) }))}
                        className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  )}

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Leverage Multiplier</label>
                    <input
                      type="number"
                      min="1"
                      max="125"
                      step="1"
                      value={tradeData.leverage}
                      onChange={(e) => setTradeData(prev => ({ ...prev, leverage: Number(e.target.value) }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Margin Mode</label>
                    <select
                      value={tradeData.margin_mode}
                      onChange={(e) => setTradeData(prev => ({ ...prev, margin_mode: e.target.value as 'cross' | 'isolated' }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    >
                      <option value="cross">Cross Margin</option>
                      <option value="isolated">Isolated Margin</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Take Profit (%)</label>
                    <input
                      type="number"
                      min="0.1"
                      max="50"
                      step="0.1"
                      value={tradeData.take_profit_percentage}
                      onChange={(e) => setTradeData(prev => ({ ...prev, take_profit_percentage: Number(e.target.value) }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">{calculateLeverageDisplay(tradeData.take_profit_percentage, tradeData.leverage, true)}</p>
                  </div>

                  <div>
                    <label className="block text-gray-300 text-sm mb-2">Stop Loss (%)</label>
                    <input
                      type="number"
                      min="0.1"
                      max="20"
                      step="0.1"
                      value={tradeData.stop_loss_percentage}
                      onChange={(e) => setTradeData(prev => ({ ...prev, stop_loss_percentage: Number(e.target.value) }))}
                      className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-indigo-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">{calculateLeverageDisplay(tradeData.stop_loss_percentage, tradeData.leverage, false)}</p>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setTradeDialogOpen(false)}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-gray-500/20 to-gray-500/20 backdrop-blur-xl border border-gray-500/50 rounded-xl text-gray-400 hover:bg-gray-500/30 transition-all duration-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleConfirmTrade}
                    className={`flex-1 px-4 py-2 backdrop-blur-xl border rounded-xl transition-all duration-300 ${
                      tradeData.side === 'buy' 
                        ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border-green-500/50 text-green-400 hover:bg-green-500/30' 
                        : 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30'
                    }`}
                  >
                    <div className="flex items-center justify-center gap-2">
                      <span>▶️</span>
                      <span>Confirm {tradeData.side === 'buy' ? 'LONG' : 'SHORT'}</span>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Confirmation Dialog */}
        {confirmDialogOpen && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="relative max-w-lg w-full">
              <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-black/90 backdrop-blur-xl border border-yellow-500/30 rounded-2xl p-6">
                <div className="flex items-center gap-3 mb-6">
                  <span className="text-3xl">⚠️</span>
                  <h3 className="text-2xl font-bold text-yellow-400">Confirm Manual Trade</h3>
                </div>
                
                <div className="relative mb-6">
                  <div className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-xl blur-lg" />
                  <div className="relative bg-black/50 backdrop-blur-xl border border-orange-500/30 rounded-xl p-4">
                    <p className="text-orange-400 mb-4">You are about to execute a manual trade. Please review all details carefully.</p>
                    
                    <div className="space-y-2 text-gray-300">
                      <h4 className="text-lg font-bold text-white mb-3">Trade Summary:</h4>
                      <p>• <strong>Coin:</strong> {selectedCoin}</p>
                      <p>• <strong>Direction:</strong> {tradeData.side === 'buy' ? 'LONG 📈' : 'SHORT 📉'}</p>
                      <p>• <strong>Order Type:</strong> {tradeData.order_type.toUpperCase()}</p>
                      <p>• <strong>Amount:</strong> {calculateTradeAmounts().tradeAmount.toFixed(2)} USDT ({tradeData.amount_percentage}%)</p>
                      <p>• <strong>Leverage:</strong> {tradeData.leverage}x ({tradeData.margin_mode})</p>
                      <p>• <strong>Take Profit:</strong> {calculateLeverageDisplay(tradeData.take_profit_percentage, tradeData.leverage, true)}</p>
                      <p>• <strong>Stop Loss:</strong> {calculateLeverageDisplay(tradeData.stop_loss_percentage, tradeData.leverage, false)}</p>
                      {tradeData.order_type === 'limit' && (
                        <p>• <strong>Limit Price:</strong> ${tradeData.limit_price?.toFixed(4)}</p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setConfirmDialogOpen(false)}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-gray-500/20 to-gray-500/20 backdrop-blur-xl border border-gray-500/50 rounded-xl text-gray-400 hover:bg-gray-500/30 transition-all duration-300"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleExecuteTrade}
                    disabled={manualTradeMutation.isPending}
                    className={`flex-1 px-4 py-2 backdrop-blur-xl border rounded-xl transition-all duration-300 disabled:opacity-50 ${
                      tradeData.side === 'buy' 
                        ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border-green-500/50 text-green-400 hover:bg-green-500/30' 
                        : 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border-red-500/50 text-red-400 hover:bg-red-500/30'
                    }`}
                  >
                    <div className="flex items-center justify-center gap-2">
                      {manualTradeMutation.isPending ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                      ) : (
                        <span>▶️</span>
                      )}
                      <span>
                        {manualTradeMutation.isPending ? 'Executing...' : `Execute ${tradeData.side.toUpperCase()}`}
                      </span>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};