import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

interface TradingStatus {
  enabled: boolean;
  auto_start: boolean;
  balance: number;
  min_balance_required: number;
  daily_loss_tracker: number;
  max_daily_loss: number;
  trades_today: number;
  last_reset_date: string;
}

interface RiskSettings {
  min_balance_required: number;
  max_daily_loss_percentage: number;
  auto_start_trading: boolean;
}

export const TradingControl: React.FC = () => {
  const [riskDialogOpen, setRiskDialogOpen] = useState(false);
  const [riskSettings, setRiskSettings] = useState<RiskSettings>({
    min_balance_required: 10,
    max_daily_loss_percentage: 5,
    auto_start_trading: true
  });
  const queryClient = useQueryClient();

  const { data: tradingStatus, isLoading } = useQuery({
    queryKey: ['trading-status'],
    queryFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/status`);
      if (!response.ok) throw new Error('Failed to fetch trading status');
      return await response.json() as TradingStatus;
    },
    refetchInterval: 5000 // Refresh every 5 seconds
  });

  const toggleTradingMutation = useMutation({
    mutationFn: async (enable: boolean) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable })
      });
      if (!response.ok) throw new Error('Failed to toggle trading');
      return response.json();
    },
    onSuccess: (data, enable) => {
      toast.success(`Trading ${enable ? 'enabled' : 'disabled'}`);
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to toggle trading');
    }
  });

  const forceStartMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/force-start`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to force start trading');
      return response.json();
    },
    onSuccess: () => {
      toast.success('🚀 Trading force-started!');
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to force start trading');
    }
  });

  const updateRiskSettingsMutation = useMutation({
    mutationFn: async (settings: RiskSettings) => {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/trading/risk-settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      if (!response.ok) throw new Error('Failed to update risk settings');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Risk settings updated');
      queryClient.invalidateQueries({ queryKey: ['trading-status'] });
    },
    onError: () => {
      toast.error('Failed to update risk settings');
    }
  });

  const handleToggleTrading = () => {
    if (tradingStatus) {
      toggleTradingMutation.mutate(!tradingStatus.enabled);
    }
  };

  const handleForceStart = () => {
    forceStartMutation.mutate();
  };

  const handleUpdateRiskSettings = () => {
    updateRiskSettingsMutation.mutate(riskSettings);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black text-white relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        </div>
        <div className="relative z-10 p-8 flex justify-center items-center min-h-screen">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-cyan-400"></div>
        </div>
      </div>
    );
  }

  if (!tradingStatus) {
    return (
      <div className="min-h-screen bg-black text-white relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        </div>
        <div className="relative z-10 p-8">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-pink-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-red-500/30 rounded-2xl p-6">
              <p className="text-red-400 font-semibold">Failed to load trading status</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const isBalanceLow = tradingStatus.balance < tradingStatus.min_balance_required;
  const isDailyLossHigh = tradingStatus.daily_loss_tracker >= tradingStatus.max_daily_loss;
  const canTrade = !isBalanceLow && !isDailyLossHigh;

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-green-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
      </div>

      <div className="relative z-10 p-8">
        {/* Header */}
        <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-cyan-500 to-blue-500 mb-8">
          🎮 Trading Control Center
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Trading Status */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-cyan-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-green-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-green-400 mb-4">Trading Status</h3>
              
              {/* Status Chips */}
              <div className="flex gap-3 mb-6">
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg ${
                  tradingStatus.enabled 
                    ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' 
                    : 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50'
                }`}>
                  <span className="text-xl">{tradingStatus.enabled ? '▶️' : '⏹️'}</span>
                  <span className={`font-semibold ${tradingStatus.enabled ? 'text-green-400' : 'text-red-400'}`}>
                    {tradingStatus.enabled ? 'ENABLED' : 'DISABLED'}
                  </span>
                </div>
                
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg ${
                  isBalanceLow 
                    ? 'bg-gradient-to-r from-red-500/20 to-pink-500/20 border border-red-500/50' 
                    : 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 border border-blue-500/50'
                }`}>
                  <span className="text-xl">💰</span>
                  <span className={`font-semibold ${isBalanceLow ? 'text-red-400' : 'text-blue-400'}`}>
                    {tradingStatus.balance.toFixed(2)} USDT
                  </span>
                </div>
              </div>

              {/* Trading Toggle */}
              <div className="flex items-center gap-4 mb-6">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tradingStatus.enabled}
                    onChange={handleToggleTrading}
                    disabled={toggleTradingMutation.isPending}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-green-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                </label>
                <span className="text-gray-300">Enable Trading</span>
              </div>

              {/* Force Start Button */}
              <button
                onClick={handleForceStart}
                disabled={forceStartMutation.isPending || tradingStatus.enabled}
                className="w-full px-6 py-3 bg-gradient-to-r from-orange-500/20 to-yellow-500/20 backdrop-blur-xl border border-orange-500/50 rounded-xl text-orange-400 hover:bg-orange-500/30 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_30px_rgba(255,165,0,0.5)] disabled:opacity-50 disabled:cursor-not-allowed mb-4"
              >
                <div className="flex items-center justify-center gap-2">
                  <span className="text-xl">🚀</span>
                  <span className="font-semibold">Force Start Trading</span>
                </div>
              </button>

              {/* Warnings */}
              {!canTrade && (
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-xl blur-lg" />
                  <div className="relative bg-black/50 backdrop-blur-xl border border-yellow-500/30 rounded-xl p-4">
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">⚠️</span>
                      <div>
                        <h4 className="text-yellow-400 font-semibold mb-1">Trading Restricted</h4>
                        <div className="text-gray-300 text-sm space-y-1">
                          {isBalanceLow && <p>Balance too low ({tradingStatus.balance.toFixed(2)} &lt; {tradingStatus.min_balance_required} USDT)</p>}
                          {isDailyLossHigh && <p>Daily loss limit exceeded ({tradingStatus.daily_loss_tracker.toFixed(2)}%)</p>}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Daily Statistics */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-xl" />
            <div className="relative bg-black/50 backdrop-blur-xl border border-blue-500/30 rounded-2xl p-6">
              <h3 className="text-2xl font-bold text-blue-400 mb-4">Daily Statistics</h3>
              
              <div className="space-y-4">
                <div>
                  <p className="text-gray-400 text-sm mb-1">Trades Today</p>
                  <p className="text-3xl font-bold text-white">{tradingStatus.trades_today}</p>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-1">Daily P&L</p>
                  <p className={`text-3xl font-bold ${
                    tradingStatus.daily_loss_tracker >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {tradingStatus.daily_loss_tracker >= 0 ? '+' : ''}{tradingStatus.daily_loss_tracker.toFixed(2)}%
                  </p>
                </div>

                <div>
                  <p className="text-gray-400 text-sm mb-1">Last Reset</p>
                  <p className="text-white">{new Date(tradingStatus.last_reset_date).toLocaleDateString()}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Management */}
        <div className="relative group mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-2xl blur-xl" />
          <div className="relative bg-black/50 backdrop-blur-xl border border-purple-500/30 rounded-2xl p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-2xl font-bold text-purple-400">Risk Management</h3>
              <button
                onClick={() => setRiskDialogOpen(!riskDialogOpen)}
                className="px-4 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300"
              >
                <div className="flex items-center gap-2">
                  <span className="text-lg">⚙️</span>
                  <span>Settings</span>
                </div>
              </button>
            </div>

            {riskDialogOpen && (
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-gray-500/10 to-gray-500/10 rounded-xl blur-lg" />
                <div className="relative bg-black/30 backdrop-blur-xl border border-gray-500/30 rounded-xl p-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div>
                      <label className="block text-gray-300 text-sm mb-2">Min Balance Required (USDT)</label>
                      <input
                        type="number"
                        value={riskSettings.min_balance_required}
                        onChange={(e) => setRiskSettings({
                          ...riskSettings,
                          min_balance_required: Number(e.target.value)
                        })}
                        className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                      />
                    </div>
                    <div>
                      <label className="block text-gray-300 text-sm mb-2">Max Daily Loss (%)</label>
                      <input
                        type="number"
                        value={riskSettings.max_daily_loss_percentage}
                        onChange={(e) => setRiskSettings({
                          ...riskSettings,
                          max_daily_loss_percentage: Number(e.target.value)
                        })}
                        className="w-full px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
                      />
                    </div>
                    <div className="flex items-center">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={riskSettings.auto_start_trading}
                          onChange={(e) => setRiskSettings({
                            ...riskSettings,
                            auto_start_trading: e.target.checked
                          })}
                          className="w-4 h-4 text-purple-600 bg-black border-gray-600 rounded focus:ring-purple-500"
                        />
                        <span className="text-gray-300">Auto-start Trading</span>
                      </label>
                    </div>
                  </div>
                  <button
                    onClick={handleUpdateRiskSettings}
                    disabled={updateRiskSettingsMutation.isPending}
                    className="px-6 py-2 bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-purple-500/50 rounded-xl text-purple-400 hover:bg-purple-500/30 transition-all duration-300 disabled:opacity-50"
                  >
                    Update Settings
                  </button>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <p className="text-gray-400 text-sm mb-1">Min Balance</p>
                <p className="text-xl font-bold text-white">{tradingStatus.min_balance_required} USDT</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm mb-1">Max Daily Loss</p>
                <p className="text-xl font-bold text-white">{tradingStatus.max_daily_loss}%</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm mb-1">Auto-start</p>
                <div className={`inline-flex items-center px-3 py-1 rounded-lg ${
                  tradingStatus.auto_start 
                    ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/50' 
                    : 'bg-gradient-to-r from-gray-500/20 to-gray-500/20 border border-gray-500/50'
                }`}>
                  <span className={`font-semibold ${tradingStatus.auto_start ? 'text-green-400' : 'text-gray-400'}`}>
                    {tradingStatus.auto_start ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};